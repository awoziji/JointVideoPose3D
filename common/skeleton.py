# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
        self._meta = {}
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def parent(self, child):
        return self._parents[child]

    def tree(self):
      for p in self._parents:
        ps = self.get_parents(p)
        print(ps)
      exit()

    def do_get_parents(self, c, t):
      p = self.parent(c)
      if c == -1 or p == -1:
        #t.append(c)
        return t
      t.append(p)
      return self.do_get_parents(p, t)

    def get_parents(self, c):
      t = [c]
      t = self.do_get_parents(c, t)
      return t

    def children(self):
        return self._children

    def children_of(self, parent):
      children = []
      for i in (range(len(self._parents))):
        if(self._parents[i] == parent):
          children.append(i)
      return children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        #print("joints_to_remove", 
              #joints_to_remove,
              #self._joints_left,
              #self._joints_right)

        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        if self._joints_left is not None:
            #print("joints left", self._joints_left)
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            #print("joints right", self._joints_right)
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right

    def meta(self):
        return self._meta
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        #print(self._has_children)
        for i, parent in enumerate(self._parents):
            #print("i", i, "p", parent)
            if parent != -1:
                self._has_children[parent] = True
            
        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
