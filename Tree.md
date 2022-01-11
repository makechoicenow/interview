# 114. 二叉树展开为链表

展开后的单链表应该与二叉树 先序遍历 顺序相同。

``` cpp
class Solution {
public:

    TreeNode* preorder(TreeNode* root){
        if (root == nullptr) {
            return nullptr;
        }

        TreeNode* tmpRight = root->right;
        root->right = preorder(root->left);
        root->left = nullptr;
        TreeNode* p = root;
        while (p->right != nullptr) {
            p = p->right;
        }
        p->right = preorder(tmpRight);
        return root;
    }

    void flatten(TreeNode* root) {
        preorder(root);
    }
};
```

# 116. 填充每个节点的下一个右侧节点指针

``` cpp
class Solution {
public:
    Node* connect(Node* root) {
        if(root == nullptr){
            return root;
        }
        vector<vector<Node*>> vecList;
        queue<Node*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            vector<Node*> tmpVec;
            tmpVec.reserve(size);
            while(size > 0){
                Node* tmp = q.front();
                q.pop();
                tmpVec.push_back(tmp);
                if (tmp->left != nullptr) {
                    q.push(tmp->left);
                }
                if (tmp->right != nullptr) {
                    q.push(tmp->right);
                }
                size--;
            }
            vecList.push_back(tmpVec);
        }
        for(auto& vec : vecList){
            for (int i = 0; i < vec.size();i++){
                if(i + 1 <= vec.size() - 1){
                    vec[i]->next = vec[i+1];
                } else {
                    vec[i]->next = nullptr;
                }
            }
        }
        return root;
    }
};
```



# 二叉树中的最大路径和
``` cpp
class Solution {
private:
    int maxSum = INT_MIN;

public:
    int maxGain(TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        
        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        int leftGain = max(maxGain(node->left), 0);
        int rightGain = max(maxGain(node->right), 0);

        // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
        int priceNewpath = node->val + leftGain + rightGain;

        // 更新答案
        maxSum = max(maxSum, priceNewpath);

        // 返回节点的最大贡献值
        return node->val + max(leftGain, rightGain);
    }

    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return maxSum;
    }
};
```

# 二叉树的最近公共祖先
``` cpp
class Solution {
public:
    TreeNode* ans;
    bool dfs(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr) return false;
        bool lson = dfs(root->left, p, q);
        bool rson = dfs(root->right, p, q);
        if ((lson && rson) || ((root->val == p->val || root->val == q->val) && (lson || rson))) {
            ans = root;
        } 
        return lson || rson || (root->val == p->val || root->val == q->val);
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root, p, q);
        return ans;
    }
};
```

# 二叉树的序列化与反序列化
``` cpp
class Codec {
public:
    void rserialize(TreeNode* root, string& str) {
        if (root == nullptr) {
            str += "None,";
        } else {
            str += to_string(root->val) + ",";
            rserialize(root->left, str);
            rserialize(root->right, str);
        }
    }

    string serialize(TreeNode* root) {
        string ret;
        rserialize(root, ret);
        return ret;
    }

    TreeNode* rdeserialize(list<string>& dataArray) {
        if (dataArray.front() == "None") {
            dataArray.erase(dataArray.begin());
            return nullptr;
        }

        TreeNode* root = new TreeNode(stoi(dataArray.front()));
        dataArray.erase(dataArray.begin());
        root->left = rdeserialize(dataArray);
        root->right = rdeserialize(dataArray);
        return root;
    }

    TreeNode* deserialize(string data) {
        list<string> dataArray;
        string str;
        for (auto& ch : data) {
            if (ch == ',') {
                dataArray.push_back(str);
                str.clear();
            } else {
                str.push_back(ch);
            }
        }
        if (!str.empty()) {
            dataArray.push_back(str);
            str.clear();
        }
        return rdeserialize(dataArray);
    }
};
```

# 105. 从前序与中序遍历序列构造二叉树
+ 定位根节点IDX, 递归
+ 106. 从中序与后序遍历序列构造二叉树 同理

``` cpp
class Solution {
public:
    int find_idx(vector<int>& vec, int s, int e, int val){
        if (s < 0 || e >= vec.size()) {
            return -1;
        }
        for (int i = s; i <= e; i++) {
            if (vec[i] == val) {
                return i;
            }
        }
        return -1;
    }
    TreeNode* build(vector<int>& preorder, int sp, int ep, vector<int>& inorder, int si, int ei){
        if (sp < 0 || sp >= preorder.size() || ep >= preorder.size() || sp > ep ) {
            return nullptr;
        }
        if (sp == ep) {
            return new TreeNode(preorder[sp]);
        }
        int root_val = preorder[sp];
        TreeNode* root = new TreeNode(root_val);
        int root_idx_in_inorder = find_idx(inorder, si, ei, root_val);
        int left_len = root_idx_in_inorder - si;
        root->left = build(preorder, sp + 1, sp + left_len, inorder, si, si + left_len -1);
        root->right = build(preorder, sp + left_len + 1, ep, inorder, si + left_len + 1, ei);
        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int p_size = preorder.size();
        int i_size = inorder.size();
        if (p_size != i_size) {
            return nullptr;
        }
        return build(preorder, 0, p_size - 1, inorder, 0, p_size - 1);
    }
};

```
# 从上到下的层序遍历
``` cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ret;
        if (root == nullptr) {
            return ret;
        }
        queue <TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int currentLevelSize = q.size();
            ret.push_back(vector<int>());
            for (int i = 1; i <= currentLevelSize; ++i) {
                auto node = q.front(); 
                q.pop();
                ret.back().push_back(node->val);
                if (node->left != nullptr){
                    q.push(node->left);
                }
                if (node->right != nullptr){
                    q.push(node->right);
                } 
            }
        }
        return ret;
    }
};
```



# 自底向上的层序遍历

``` cpp
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        auto levelOrder = vector<vector<int>>();
        if (!root) {
            return levelOrder;
        }
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            auto level = vector<int>();
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                auto node = q.front();
                q.pop();
                level.push_back(node->val);
                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
            }
            levelOrder.push_back(level);
        }
        reverse(levelOrder.begin(), levelOrder.end());
        return levelOrder;
    }
};

```

# 二叉树的z字形遍历
``` cpp
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (!root) {
            return ans;
        }

        queue<TreeNode*> nodeQueue;
        nodeQueue.push(root);
        bool isOrderLeft = true;

        while (!nodeQueue.empty()) {
            deque<int> levelList;
            int size = nodeQueue.size();
            for (int i = 0; i < size; ++i) {
                auto node = nodeQueue.front();
                nodeQueue.pop();
                if (isOrderLeft) {
                    levelList.push_back(node->val);
                } else {
                    levelList.push_front(node->val);
                }
                if (node->left) {
                    nodeQueue.push(node->left);
                }
                if (node->right) {
                    nodeQueue.push(node->right);
                }
            }
            ans.emplace_back(vector<int>{levelList.begin(), levelList.end()});
            isOrderLeft = !isOrderLeft;
        }

        return ans;
    }
};
```


# 二叉树的垂序遍历
``` cpp
class Solution {
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<tuple<int, int, int>> nodes;

        function<void(TreeNode*, int, int)> dfs = [&](TreeNode* node, int row, int col) {
            if (!node) {
                return;
            }
            nodes.emplace_back(col, row, node->val);
            dfs(node->left, row + 1, col - 1);
            dfs(node->right, row + 1, col + 1);
        };

        dfs(root, 0, 0);
        sort(nodes.begin(), nodes.end());
        vector<vector<int>> ans;
        int lastcol = INT_MIN;
        for (const auto& [col, row, value]: nodes) {
            if (col != lastcol) {
                lastcol = col;
                ans.emplace_back();
            }
            ans.back().push_back(value);
        }
        return ans;
    }
};
```

# 二插搜索树的插入
``` cpp
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return new TreeNode(val);
        }
        TreeNode* pos = root;
        while (pos != nullptr) {
            if (val < pos->val) {
                if (pos->left == nullptr) {
                    pos->left = new TreeNode(val);
                    break;
                } else {
                    pos = pos->left;
                }
            } else {
                if (pos->right == nullptr) {
                    pos->right = new TreeNode(val);
                    break;
                } else {
                    pos = pos->right;
                }
            }
        }
        return root;
    }
};

```

# 450. 删除二叉搜索树中的节点
``` cpp
class Solution {
public:
    TreeNode* find_min_node_in_bst(TreeNode* node) {
        if (node == nullptr) {
            return nullptr;
        }
        while(node->left != nullptr) {
            node = node->left;
        }
        return node;
    }


    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) {
            return nullptr;
        }
        if (root->val > key) {
            root->left = deleteNode(root->left, key);
        } else if (root->val < key) {
            root->right = deleteNode(root->right, key);
        } else {
            if(root->right == nullptr){
                root = root->left;
            } else if(root->left == nullptr) {
                root = root->right;
            } else {
                TreeNode* newRoot = find_min_node_in_bst(root->right);
                root->val = newRoot->val;
                // 一定得用到返回值，即root->right = XXX
                root->right = deleteNode(root->right, newRoot->val);
            }
            
        }
        return root;
    }
};
```

# 98. 验证二叉搜索树

+ BST 中序遍历有序

``` cpp
class Solution {
private:
    int cur = INT_MIN;
    bool is_start = false;
    bool res = true;
public:
    
    void inorder(TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        if (res == false) {
            return;
        }
        inorder(node->left);
        if (is_start && cur >= node->val) {
            res = false;
            return;
        } else {
            cur = node->val;
            if(is_start == false) {
                is_start = true;
            }
        }
        inorder(node->right);
    }
    bool isValidBST(TreeNode* root) {
        if (root == nullptr) {
            return true;
        }
        inorder(root);
        return res;
    }
};
```
# 701. 二叉搜索树中的插入操作


``` cpp
class Solution {
private:
    bool isInserted = false;
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return new TreeNode(val);
        }
        if (val == root->val) {
            isInserted = true;
            return root;
        }
        if (isInserted == true) {
            return root;
        }
        if (root->val > val) {
            if(root->left == nullptr){
                root->left = new TreeNode(val);
                isInserted = true;
            } else {
                root->left = insertIntoBST(root->left, val);
            }
        }

        if (root->val < val) {
            if (root->right == nullptr){
                root->right = new TreeNode(val);
                isInserted = true;
            } else {
                root->right = insertIntoBST(root->right, val);
            }
        }
        return root;
    }
};
```

# 230. 二叉搜索树中第K小的元素
``` cpp
class Solution {
private:
    int rank = 0;
    int res = INT_MIN;
public:
    void preorder(TreeNode* root, int k){
        if (root == nullptr || rank == k) {
            return;
        }
        preorder(root->left, k);
        rank++;
        if (rank == k) {
            res = root->val;
            return;
        }
        preorder(root->right, k);
    }
    int kthSmallest(TreeNode* root, int k) {
        preorder(root, k);
        return res;
    }
};
```

# 538/1038. 把二叉搜索树转换为累加树
``` cpp
class Solution {
private:
    int sum = 0;
public:
    void inorder(TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        inorder(node->right);
        sum += node->val;
        node->val = sum;
        inorder(node->left);
    }
    TreeNode* bstToGst(TreeNode* root) {
        inorder(root);
        return root;
    }
};
```

# 不同的二叉搜索树
``` cpp
class Solution {
public:
    int numTrees(int n) {
        vector<int> G(n + 1, 0);
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }
};

```

# 700. 二叉搜索树中的搜索
``` cpp
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return nullptr;
        }
        if (root->val == val) {
            return root;
        }
        TreeNode* res = nullptr;
        if (root->val > val) {
            res = searchBST(root->left, val);
        }
        if(root->val < val) {
            res = searchBST(root->right, val);
        }
        return res;
    }
};
```

# 95. 不同的二叉搜索树 II (打印出全部的树)

``` cpp
class Solution {
private:
    vector<TreeNode*> res;
public:
    vector<TreeNode*> dfs(int s, int e){
        if(s > e){
            return {};
        }
        if(s == e){
            vector<TreeNode*> vec;
            vec.push_back(new TreeNode(s));
            return vec;
        }
        vector<TreeNode*> tmpRes;
        for (int i = s; i <= e; i++) {
            vector<TreeNode*> lTrees = dfs(s, i - 1);
            vector<TreeNode*> rTrees = dfs(i + 1, e);
            
            if(lTrees.empty()){
                lTrees.push_back(nullptr);
            }
            if(rTrees.empty()){
                rTrees.push_back(nullptr);
            }

            for(int j = 0; j < lTrees.size(); j++) {
                for (int k = 0; k < rTrees.size(); k++) {
                    TreeNode* root = new TreeNode(i);
                    if(lTrees[j] != nullptr){
                        root->left = lTrees[j];
                    }
                    if(rTrees[k] != nullptr) {
                        root->right = rTrees[k];
                    }
                    tmpRes.push_back(root);
                }
            }
        }
        return tmpRes;
    }
    
    vector<TreeNode*> generateTrees(int n) {
        return dfs(1, n);
    }
};
```

# 652. 寻找重复的子树

``` cpp
class Solution {
private:
    unordered_map<string, int> record;
    vector<TreeNode*> res;
public:

    string find(TreeNode* node){
        if (node == nullptr) {
            return "#";
        }
        string lStr = find(node->left);
        string rStr = find(node->right);
        //这个相加的顺序很重要
        string root_str = to_string(node->val) +  ","  + lStr + "," + rStr;
        record[root_str]++;
        if( record[root_str] == 2){
            res.push_back(node);
        }
        
        return root_str;

    }
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        find(root);
        return res;
    }
};
```

# 654. 最大二叉树
给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。

``` cpp
class Solution {
public:
    int find_max_idx(vector<int>& nums, int s, int e) {
        if (s <0 || e >= nums.size() || s > e){
            return -1;
        }
        int max = INT_MIN;
        int max_idx = -1;
        for (int i = s; i <= e; i++) {
            if (nums[i] > max) {
                max = nums[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    TreeNode* constructMaximumBinaryTree(vector<int>& nums, int s, int e) {
        int root_idx = find_max_idx(nums, s, e);
        if(root_idx == -1){
            return nullptr;
        }
        TreeNode* root = new TreeNode(nums[root_idx]);
        TreeNode* left = constructMaximumBinaryTree(nums, s, root_idx - 1);
        TreeNode* right = constructMaximumBinaryTree(nums, root_idx + 1, e);
        root->left = left;
        root->right = right;
        return root;
    }


    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return constructMaximumBinaryTree(nums, 0, nums.size()-1);
    }
};
```

