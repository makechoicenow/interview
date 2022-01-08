# 深度优先搜索

## 模板
``` C++

/*
* dfs 模板
* @param[in] iutput 输入数据指针
* @param[inout] cur or gap 标记当前位置距离目标的距离
* @param[out]  path 当前路径，也就是中间结果
* @param[out]  result 存放最终结果
* @return 路径长度，如果是求路径本身，则不需要返回长度
*/

void dfs(type *input, type *path, int cur or gap, type* result){
	if (数据非法) return 0; //终止函数
	if (cur == input.size() or gap == 0){ //收敛条件
		将path放入result
	}
	if(可以剪枝) return; 
	for(......){//执行所有可能的扩展动作
		修改path, 执行动作
		dfs(input, cur+1 or gap-1, result);
		恢复path
	}
}
```

## 例子1 岛屿数量



``` cpp
class Solution {
private:
    int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    vector<vector<bool>> memo;
    int res;
    int m;
    int n;
public:
    void dfs(vector<vector<char>>& grid, int x, int y){
        if(grid[x][y] == '1'){
            grid[x][y] = '#';
        }
        for(int i = 0;i<4;i++){
            int x1 = x + dirs[i][0];
            int y1 = y + dirs[i][1];
            if(x1 >=0 && x1 < m && y1>=0 && y1 < n && grid[x1][y1]=='1'){
                dfs(grid, x1, y1);
            }
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        m = grid.size();
        n = grid[0].size();
        memo = vector<vector<bool>>(m, vector<bool>(n));
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(grid[i][j] == '1'){
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }
};
```

## 例子2 分割回文串

示例 1：

输入：s = "aab"

输出：[["a","a","b"],["aa","b"]]

示例 2：

输入：s = "a"

输出：[["a"]]

``` cpp
class Solution {
private:
    vector<vector<bool>> dp;
    vector<vector<string>> res;
    vector<string> tmpAns;
    int n = 0;

public:
    void dfs(const string& s, int idx){
        if(idx == n){
            res.push_back(tmpAns);
            return;
        }
        for(int i = idx;i<n;i++){
            if(dp[idx][i]){
                tmpAns.push_back(s.substr(idx, i - idx + 1));
                dfs(s,i+1);
                tmpAns.pop_back();
            }
        } 
    }
    vector<vector<string>> partition(string s) {
        if (s.empty()) {
            return res;
        }
        n = s.size();
        dp.assign(n, vector<bool>(n, false));
        for(int i=0; i<n;i++){
            for(int j=i;j>=0;j--){
                dp[i][j]=true;
            }
        }
        for(int i=n-1;i>=0;i--){
            for(int j=i+1;j<n;j++){
                dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1];
            }
        }
        dfs(s, 0);
        return res;
    }
};
```

# 广度优先搜索


## 模板
 
``` cpp

// 需要一个队列，用于一层一层扩展
// 一个hashset, 用于判重
// 一棵树，(只求长度是不需要)用于存储所有路径   可以用unordered_map实现

int bfs(Node start, Node target){
	Queue<Node> queue;
	HashSet<Node> visited;

	q.push(start);//将起点计入队列
	visited.add(start);
	int step = 0;//记录扩散的步数
	
	while(q.empty() == false){
		int size = q.size();
		/*将当前队列中所有的节点向四周扩散*/
		for(int i = 0; i < size; i++){
			Node tmp = q.front();
			q.pop();
			/*划重点：这里怕近端是否到达终点*/
			if ( tmp == target){
				return step;
			}
			/*将tmp的相邻节点加入队列*/
			for (Node node : tmp.adj()){
				if (visited.count(node) == 0) {
					q.push(node);
					visited.add(node);
				}
			}
		}
		/*遍历完这一步所有的节点后，更新步数*/
		step++;
	}
}

```

## 例子1 单词接龙

给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。


``` cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        if(beginWord.size() != endWord.size()){
            return 0;
        }

        queue<string> q;
        unordered_set<string> set;
        unordered_set<string> visited;
        for(string& s : wordList){
            set.insert(s);
        }

        int step = 1;
        q.push(beginWord);
        visited.insert(beginWord);
        while(!q.empty()){
            int sz = q.size();
            for(int i = 0; i < sz;i++){
                string tmp = q.front();
                q.pop();
                for(int j = 0; j<tmp.size();j++){
                    for(char c = 'a';c<='z';c++){
                        if(c == tmp[j]){
                            continue;
                        }
                        string tmp_copy = tmp;
                        tmp_copy[j] = c;
                        if(!set.count(tmp_copy) || visited.count(tmp_copy) != 0){
                            continue;
                        }
                        if(tmp_copy == endWord){
                            return step + 1;
                        }
                        q.push(tmp_copy);
                        visited.insert(tmp_copy);
                    }
                }
            }
            step++;            
        }
        return 0;
    }
};
```


# 拓扑排序



## 模板 + 例子1  课程表

``` cpp
class Solution {

public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        //建图
        vector<vector<int>> edges;
        vector<int> indegs;
        edges.resize(numCourses);
        indegs.resize(numCourses);
        for (auto& vec : prerequisites) {
            edges[vec[1]].push_back(vec[0]);
            indegs[vec[0]]++;
        }

        //将所有入度为0的节点入队
        queue<int> q;
        for(int i = 0; i < numCourses;i++){
            if(indegs[i] == 0){
                q.push(i);
            }
        }
        
        vector<int> res;
        int visited = 0;
        while (!q.empty()) {
            //从任何一个入度为0的节点开始遍历都是可以的  
            int tmp = q.front();
            q.pop();
            visited++;
            res.push_back(tmp);
            auto& vec = edges[tmp];
            for (int v : vec) {
                indegs[v]--;
                //如果减一后入度变为0，说明该节点也可以作为下一个排序, 入队
                if (indegs[v] == 0) {
                    q.push(v);
                }
            }
        }
        //所有节点都可以被遍历到
        if (visited != numCourses) {
            res.clear();
        }
        return res;

    }
};
```



# 堆排序
### 平均时间 O(NlogN), 最坏时间 O(NlogN), 空间复杂度 O(1)
## 模板+例子1 链表排序

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */

class Solution {

private:
    vector<ListNode*> heap;
public:

    void minHeapAdjust(vector<ListNode*>& heap, int idx, int heapSize){
        int l = 2*idx+1;
        int r = 2*idx+2;
	//用max作为变量，保证min是当前节点和其左右孩子之间最大的，也就是应该作为遍历临时根节点
        int max = idx;
        if( l < heapSize && heap[l]->val < heap[max]->val){
            min = l;
        }
        if( r < h eapSize && heap[r]->val < heap[max]->val){
           max = r;
        }
        if (max!= idx) {
            swap(heap[max],heap[idx]);
            minHeapAdjust(heap,max, heapSize);
        }
    }

    void buildMinHeap(vector<ListNode*>& heap, int heapSize){
	//建堆: 从数组的中间开始逆序调整
        for(int i=heapSize/2;i>=0;i--) {
            minHeapAdjust(heap, i, heapSize);
        }
    }
    ListNode* sortList(ListNode* head) {
        if(head == nullptr || head->next == nullptr){
            return head;
        }
        while(head != nullptr){
            cout<<head<<endl;
            heap.push_back(head);
            head = head->next;
        }
        int heapSize = heap.size();

        buildMinHeap(heap, heapSize);


        ListNode* node = new ListNode(0);
        ListNode* p = node;
	/*建好堆后的排序：因为第一个元素是最大的，所以交换的末尾。
	接着重新调整堆，使第二大的元素到堆顶。
	所以应该排除最大的元素，因此堆的大小需要减1后调整。*/
        for(int i=heap.size()-1;i>=0;i--){
            p->next = heap[0];
            p = p->next;
            swap(heap[0], heap[i]);
            --heapSize;
            minHeapAdjust(heap, 0, heapSize);
        }
        p->next=nullptr;
        ListNode* res = node->next;
        delete node;
        return res;

    }
};
```

## 例子二 数据流的中位数

``` cpp
class MedianFinder {
private:
/*
priority_queue<Type, Container, Functional>
Type为数据类型， Container为保存数据的容器，Functional为元素比较方式。
如果不写后两个参数，那么容器默认用的是vector，比较方式默认用operator<，也就是优先队列是大顶堆，队头元素最大。
*/
  priority_queue<int, vector<int>, less<int>> h1; //大顶堆
  priority_queue<int, vector<int>, greater<int>> h2;//小顶堆
public:
    MedianFinder() {
    }
    
    void addNum(int num) {
        if (h1.size() == 0 || num <= h1.top()) {
            h1.push(num);
            if (h2.size() + 1 < h1.size()) {
                h2.push(h1.top());
                h1.pop();
            }
        }else{
            h2.push(num);
            if(h2.size()>h1.size()){
                h1.push(h2.top());
                h2.pop();
            }
        }
    }
    
    double findMedian() {
        int totalSize = h1.size() + h2.size();
        if (totalSize % 2 == 0) {
            return (double)((h1.top()+h2.top())/2.0);
        } else {
            return (double)(h1.top());
        }
    }
};

```

# 快速排序

### 平均时间复杂度 O(NlogN), 最坏情况 N^2, 空间复杂度 O(logN)

## 模板和实现
``` cpp
class Solution {
public:

    int partition(vector<int>& nums, int low, int high) {
        int pivot = nums[low];
        int left = low;
        int right = high;
        while (left < right) {
            //重点：从右侧开始
            while (left < right && nums[right] >= pivot) {
                right--;
            }
            while (left < right && nums[left] <= pivot) {
                left++;
            }
            swap(nums[left], nums[right]);
        }
        swap(nums[low], nums[left]);
        return left;
    }

    void quickSort(vector<int>& nums, int left, int right){
        if (left >= right) {
            return;
        }
        int mid = partition(nums, left, right);
        quickSort(nums, left, mid - 1);
        quickSort(nums, mid + 1, right);
    }
    vector<int> sortArray(vector<int>& nums) {
        quickSort(nums, 0, nums.size() - 1);
        return nums;
    }
};
```

# 翻转链表

## 实现

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverse(ListNode* s, ListNode* e){
        ListNode* pre = nullptr;
        ListNode* cur = s;
        ListNode* nxt = s;

        while(cur != e){
            nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        } 
        return pre;
    }

    // 翻转整个链表
    ListNode* reverseList(ListNode* head) {
        return reverse(head, nullptr);
    }

    /*
    * k个一组反转链表，不足k个保持不变
    */
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (head == nullptr) {
            return head;
        }
        ListNode* s = head;
        ListNode* e = head;
        for(int i = 0; i < k; i++){
            if (e == nullptr){
                return head;
            }
            e = e->next;
        }
        ListNode* newHead = reverse(s ,e);
        s->next = reverseKGroup(e, k);
        return newHead;
    }

    /*
    * 给你单链表的头指针 head 和两个整数 left 和 right ,其中 left <= right 。
    * 请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表
    */
    ListNode *reverseBetween(ListNode *head, int left, int right) {
        // 因为头节点有可能发生变化，使用虚拟头节点可以避免复杂的分类讨论
        ListNode *dummyNode = new ListNode(-1);
        dummyNode->next = head;

        ListNode *pre = dummyNode;
        // 第 1 步：从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
        // 建议写在 for 循环里，语义清晰
        for (int i = 0; i < left - 1; i++) {
            pre = pre->next;
        }

        // 第 2 步：从 pre 再走 right - left + 1 步，来到 right 节点
        ListNode *rightNode = pre;
        for (int i = 0; i < right - left + 1; i++) {
            rightNode = rightNode->next;
        }

        // 第 3 步：切断出一个子链表（截取链表）
        ListNode *leftNode = pre->next;
        ListNode *curr = rightNode->next;

        // 注意：切断链接
        pre->next = nullptr;
        rightNode->next = nullptr;

        // 第 4 步：同第 206 题，反转链表的子区间
        reverse(leftNode, nullptr);

        // 第 5 步：接回到原来的链表中
        pre->next = rightNode;
        leftNode->next = curr;
        return dummyNode->next;
    }
};
```

# Trie树
## 基本实现
``` cpp
class Trie {
private:
  vector<Trie*> childs;
  bool isEnd;
public:
    Trie() : childs(26), isEnd(false) {

    }
    
    void insert(string word) {
        Trie* node = this;
        for(char ch : word){
            if(node->childs[ch-'a'] == nullptr){
                node->childs[ch-'a'] = new Trie();
            }
            node = node->childs[ch-'a'];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Trie* node = this;
        for(char ch : word){
            if(node->childs[ch-'a'] == nullptr){
                return false;
            }
            node = node->childs[ch-'a'];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Trie* node = this;
        for(char ch : prefix){
            if(node->childs[ch-'a'] == nullptr){
                return false;
            }
            node = node->childs[ch-'a'];
        }
        return node!=nullptr;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */


```

## 单词搜索： DFS+Trie
``` cpp
struct TrieNode {
    string word;
    unordered_map<char,TrieNode *> children;
    TrieNode() {
        this->word = "";
    }   
};

void insertTrie(TrieNode * root,const string & word) {
    TrieNode * node = root;
    for (auto c : word){
        if (!node->children.count(c)) {
            node->children[c] = new TrieNode();
        }
        node = node->children[c];
    }
    node->word = word;
}

class Solution {
private:
    int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    unordered_set<string> res;
    int m;
    int n;
public:
    void dfs(vector<vector<char>>& board, int x, int y, TrieNode* trie){
        char ch = board[x][y];
        if(trie->children.find(ch)==trie->children.end()){
            return;
        }
        trie = trie->children[ch];
        if(trie->word.size()>0){
            res.insert(trie->word);
        }
        board[x][y] = '#';
        for(int i = 0;i < 4; i++){
            int nx = x + dirs[i][0];
            int ny = y + dirs[i][1];
            if(nx >= 0 && nx < m && ny>=0 && ny<n){
                if(board[nx][ny] != '#'){
                    dfs(board, nx, ny, trie);
                }
            }
        }
        board[x][y] = ch;
    }
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode* trie = new TrieNode();
        m = board.size();
        n = board[0].size();
        for(string w : words){
            insertTrie(trie, w);
        }
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                dfs(board, i, j, trie);
            }
        }
        vector<string> finalRes;
        for(string s : res){
            finalRes.emplace_back(s);
        }
        return finalRes;
    }
};
```


# 二分查找

## 模板
``` cpp
int binarySearch(const vector<int>& nums, int s, int e, int t){
    if ( s >= nums.size()){
        return -1;
    }
    int l = s;
    int r = e;
    while (l + 1 < r) {
        int mid = l + (r - l)/2;
        if (nums[mid] == t) {
            return mid;
        } else if (nums[mid] > t) {
            r = mid;
        } else if (nums[mid] < t) {
            l = mid;
        }
    }
    if (nums[l] == t) {
        return l;
    }
    if (nums[r] == t) {
        return r;
    }
    return -1;
}
```

## 例子1 搜索旋转排序数组
``` cpp
class Solution {
public:
    int binarySearch(const vector<int>& nums, int s, int e, int t){
        if ( s >= nums.size()){
            return -1;
        }
        int l = s;
        int r = e;
        while (l + 1 < r) {
            int mid = l + (r - l)/2;
            if (nums[mid] == t){
                return mid;
            } else if (nums[mid] > t) {
                r = mid;
            } else if (nums[mid] < t) {
                l = mid;
            }
        }
        if(nums[l] == t){
            return l;
        }
        if(nums[r] == t){
            return r;
        }
        return -1;
    }
    int search(vector<int>& nums, int target) {
        int idx = 0;
        while (idx < nums.size()) {
            if (idx + 1 >= nums.size()) {
                break;
            }
            if(nums[idx] >= nums[idx+1]){
                break;
            }
            idx++;
        }

        int lf = binarySearch(nums, 0, idx, target);
        if (lf >= 0) {
            return lf;
        }
        int rf = binarySearch(nums, idx+1, nums.size()-1, target);
        if (rf >= 0){
            return rf;
        }
        return -1;
    }
};
```


# 二叉树的遍历

## 先序遍历
``` cpp
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        if(!root) return {};
        vector<int> res;
        stack<TreeNode*> stk;
        stk.push(root);
        while (stk.size() > 0) {
            TreeNode* tmp = stk.top();
            stk.pop();
            res.push_back(tmp->val);
            if (tmp->right != nullptr) {
                stk.push(tmp->right);
            }
            if (tmp->left != nullptr) {
                stk.push(tmp->left);
            }
        }
        return res;
    }
};
```

## 中序遍历
``` cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root) return {};
        stack<TreeNode*> stk;
        TreeNode* p = root;
        vector<int> res;
        while (stk.size() != 0 || p != nullptr ) {
            while (p != nullptr) {
                stk.push(p);
                p = p->left;
            }
            TreeNode* tmp = stk.top();
            stk.pop();
            res.push_back(tmp->val);
            if (tmp->right != nullptr) {
                p = tmp->right;
            }
        }
        return res;
    }
};
```

## 后序遍历
``` cpp
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        if(root == nullptr) return {};
        stack<TreeNode*> stk;
        vector<int> res;
        TreeNode* p = root;
        TreeNode* pre = nullptr;
        while (stk.size() > 0 || p != nullptr) {
            while (p != nullptr) {
                stk.push(p);
                p = p->left;
            }
            TreeNode* tmp = stk.top();
            stk.pop();
            //两种情况可以输出
            //1. 右节点刚刚输出，需要用一个节点缓存上一个输出的节点.
            //2. 右节点为空, 因为入栈的方式，左右节点此时都为空，即为叶子节点.
            //输出后，相当于一个子树遍历完成，所有p置为nullptr
            if (tmp->right == pre || tmp->right == nullptr) {
                res.push_back(tmp->val);
                pre = tmp;
                p = nullptr;
            } else {
                //要先遍历右子树，暂时不能输出根节点，所有再入栈
                stk.push(tmp);
                p = tmp->right;
            }
        }
        return res;
    }
};
```

#  栈

## 例子1.计算器
``` cpp
class Solution {
public:
    int calculate(string s) {
        stack<int> stk;
        char sign = '+';
        int num = 0;
        //将一切转换为加法
        for (int i=0; i < s.size(); i++) {    
            char c = s[i];      
            if (isdigit(c)) {
                num = num * 10 + int(c - '0');
            }
            if ( (!isdigit(c) && c != ' ') || i == s.size()-1 ) {
                int tmp = 0;
                switch (sign) {
                    case '+':
                        stk.push(num);
                        break;
                    case '-':
                        stk.push(-num);
                        break;
                    case '*':
                        tmp = stk.top();
                        stk.pop();
                        stk.push(tmp * num);
                        break;
                    case '/':
                        stk.top()/=num;
                        break;
                }
                sign = c;
                num = 0;
            }   
        }

        int sum = 0;
        while (!stk.empty()) {
            sum+=stk.top();
            stk.pop();
        }
        return sum;
    }
};
```



