# LRU Cache

``` cpp
struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity): capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }
    
    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};
```

# 线程安全的阻塞队列
``` cpp
#ifndef BLOCKINGQUEUE_H
#define BLOCKINGQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <assert.h>

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue()
        :m_mutex(),
          m_condition(),
          m_data()
    {
    }

    // 禁止拷贝构造
    BlockingQueue(BlockingQueue&) = delete;

    ~BlockingQueue()
    {
    }

    void push(T&& value)
    {
        // 往队列中塞数据前要先加锁
        std::unique_lock<std::mutex> lock(m_mutex);
        m_data.push(value);
        m_condition.notify_all();
    }

    void push(const T& value)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_data.push(value);
        m_condition.notify_all();
    }

    T take()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while(m_data.empty())
        {
            m_condition.wait(lock);
        }
        assert(!m_data.empty());
        T value(std::move(m_data.front()));
        m_data.pop();

        return value;
    }

    size_t size() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_data.size();
    }
private:
    // 实际使用的数据结构队列
    std::queue<T> m_data;

    // 条件变量的锁
    std::mutex m_mutex;
    std::condition_variable m_condition;
};
#endif // BLOCKINGQUEUE_H

```

# 线程池
``` cpp

#ifndef ALTAR_THREAD_POOL_H_
#define ALTAR_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "trpc/common/logging/trpc_logging.h"
#include "trpc/log/trpc_log.h"

namespace altar {
class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

  size_t get_queue_size();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()> > tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty()) return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }
        task();
      }
      TRPC_FMT_WARN("thread_pool worker done {}", std::this_thread::get_id());
      return;
    });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  TRPC_FMT_WARN("thread_pool Stop Start {}", std::this_thread::get_id());
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) {
    TRPC_FMT_WARN("thread_pool worker start join {}", worker.get_id());
    worker.join();
  }
  TRPC_FMT_WARN("thread_pool Stop Done {}", std::this_thread::get_id());
}

// get_queu_size
inline size_t ThreadPool::get_queue_size() { return this->tasks.size(); }

};  // namespace altar
#endif


```


# 智能指针
``` cpp
#include <utility>  // std::swap

class shared_count {
 public:
  shared_count() noexcept : count_(1) {}
  void add_count() noexcept { ++count_; }
  long reduce_count() noexcept { return --count_; }
  long get_count() const noexcept { return count_; }

 private:
  long count_;
};

template <typename T>
class smart_ptr {
 public:
  template <typename U>
  friend class smart_ptr;

  explicit smart_ptr(T* ptr = nullptr) : ptr_(ptr) {
    if (ptr) {
      shared_count_ = new shared_count();
    }
  }
  ~smart_ptr() {
    if (ptr_ && !shared_count_->reduce_count()) {
      delete ptr_;
      delete shared_count_;
    }
  }

  smart_ptr(const smart_ptr& other) {
    ptr_ = other.ptr_;
    if (ptr_) {
      other.shared_count_->add_count();
      shared_count_ = other.shared_count_;
    }
  }
  template <typename U>
  smart_ptr(const smart_ptr<U>& other) noexcept {
    ptr_ = other.ptr_;
    if (ptr_) {
      other.shared_count_->add_count();
      shared_count_ = other.shared_count_;
    }
  }
  template <typename U>
  smart_ptr(smart_ptr<U>&& other) noexcept {
    ptr_ = other.ptr_;
    if (ptr_) {
      shared_count_ = other.shared_count_;
      other.ptr_ = nullptr;
    }
  }
  template <typename U>
  smart_ptr(const smart_ptr<U>& other, T* ptr) noexcept {
    ptr_ = ptr;
    if (ptr_) {
      other.shared_count_->add_count();
      shared_count_ = other.shared_count_;
    }
  }
  smart_ptr& operator=(smart_ptr rhs) noexcept {
    rhs.swap(*this);
    return *this;
  }

  T* get() const noexcept { return ptr_; }
  long use_count() const noexcept {
    if (ptr_) {
      return shared_count_->get_count();
    } else {
      return 0;
    }
  }
  void swap(smart_ptr& rhs) noexcept {
    using std::swap;
    swap(ptr_, rhs.ptr_);
    swap(shared_count_, rhs.shared_count_);
  }

  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  operator bool() const noexcept { return ptr_; }

 private:
  T* ptr_;
  shared_count* shared_count_;
};

template <typename T>
void swap(smart_ptr<T>& lhs, smart_ptr<T>& rhs) noexcept {
  lhs.swap(rhs);
}

template <typename T, typename U>
smart_ptr<T> static_pointer_cast(const smart_ptr<U>& other) noexcept {
  T* ptr = static_cast<T*>(other.get());
  return smart_ptr<T>(other, ptr);
}

template <typename T, typename U>
smart_ptr<T> reinterpret_pointer_cast(const smart_ptr<U>& other) noexcept {
  T* ptr = reinterpret_cast<T*>(other.get());
  return smart_ptr<T>(other, ptr);
}

template <typename T, typename U>
smart_ptr<T> const_pointer_cast(const smart_ptr<U>& other) noexcept {
  T* ptr = const_cast<T*>(other.get());
  return smart_ptr<T>(other, ptr);
}

template <typename T, typename U>
smart_ptr<T> dynamic_pointer_cast(const smart_ptr<U>& other) noexcept {
  T* ptr = dynamic_cast<T*>(other.get());
  return smart_ptr<T>(other, ptr);
}
```

# 跳表
``` cpp
struct SkipListNode {
	int val;
	vector<SkipListNode *> level;
	SkipListNode (int _val, int sz=32) : val(_val), level(sz, nullptr) {}
};
class Skiplist {
private:
    SkipListNode *head, *tail;
    int level, length;
public:
	static constexpr int MAXL = 32;
    static constexpr int P = 4;
    static constexpr int S = 0xFFFF;
    static constexpr int PS = S / 4;

    Skiplist() {
        level = length = 0;
        tail = new SkipListNode(INT_MAX, 0);
        head = new SkipListNode(INT_MAX);
        for (int i = 0; i < MAXL; ++i) { 
        	head->level[i] = tail;
        }
    }

    SkipListNode* find(int val) {
        SkipListNode *p = head;
        for (int i = level - 1; i >= 0; --i) {
            while (p->level[i] && p->level[i]->val < val) {
                p = p->level[i];
            }
        }
        p = p->level[0];
        return p;
    }
    
    bool search(int target) {
        SkipListNode *p = find(target);
        return p->val == target;
    }
    
    void add(int val) {
        vector<SkipListNode *> update(MAXL);
        SkipListNode *p = head;
        for (int i = level - 1; i >= 0; --i) {
            while (p->level[i] && p->level[i]->val < val) {
                p = p->level[i];
            }
            update[i] = p;
        }
        int lv = randomLevel();
        if (lv > level) {
            lv = ++level;
            update[lv - 1] = head; 
        }
        SkipListNode *newNode = new SkipListNode(val, lv);
        for (int i = lv - 1; i >= 0; --i) {
            p = update[i];
            newNode->level[i] = p->level[i];
            p->level[i] = newNode;
        }
        ++length;
    }
    
    bool erase(int val) {
        vector<SkipListNode *> update(MAXL + 1);
        SkipListNode *p = head;
        for (int i = level - 1; i >= 0; --i) {
            while (p->level[i] && p->level[i]->val < val) {
                p = p->level[i];
            }
            update[i] = p;
        }
        p = p->level[0];
        if (p->val != val) return false;
        for (int i = 0; i < level; ++i) {
            if (update[i]->level[i] != p) {
                break;
            }
            update[i]->level[i] = p->level[i];
        }
        while (level > 0 && head->level[level - 1] == tail) --level;
        --length;
        return true;
    }

    int randomLevel() {
        int lv = 1;
        while (lv < MAXL && (rand() & S) < PS) ++lv;
        return lv;
    }
};
```

# 文件系统
``` cpp
class Trie{
public:
    unordered_map<string,Trie*>s2children;
    bool isEnd;
    int val;
    Trie(){
        isEnd=false;
        val=-1;
    }
    bool insert(vector<string>&words,int val){
        Trie*root=this;
        int n=words.size();
        for(int i=0;i<n-1;i++){
            if(root->s2children[words[i]]==nullptr)return false;
            root=root->s2children[words[i]];
        }
        if(root->s2children[words[n-1]]!=nullptr)return false;
        root->s2children[words[n-1]]=new Trie();
        root=root->s2children[words[n-1]];
        root->isEnd=true;
        root->val=val;
        return true;
    }
    int query(vector<string>&words){
        Trie*root=this;
        for(auto&s:words){
            if(root->s2children[s]==nullptr)return -1;
            root=root->s2children[s];
        }
        if(root->isEnd==false)return -1;
        return root->val;
    }
};
class FileSystem {
public:
    Trie trie;
    FileSystem() {
    }
    bool createPath(string path, int value) {
        vector<string>paths=split(path,"/");
        return trie.insert(paths,value);
    }
    int get(string path) {
        vector<string>paths=split(path,"/");
        return trie.query(paths);
    }
    vector<string>split(string&s,string c){
        vector<string>ans;
        int i=1,j;
        while((j=s.find(c,i))!=-1){
            ans.push_back(s.substr(i,j-i));
            i=j+c.size();
        }
        if(i!=s.size()){
            ans.push_back(s.substr(i));
        }
        return ans;
    }
};
```

# 大整数计算