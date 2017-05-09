class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution(object):
    res = [[]]
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        levelorder(root,0)
        
    def levelorder(self, root, level):
        if len(res) < level + 1:
            res.append([])
        if root.left != None:
            out[level + 1].append(root.left.val)
        if root.left != Nono:
            root[level + 1].append(root.right.val)
        self.levelorder(root,level+1)
    