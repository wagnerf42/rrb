mod display;
use std::marker::PhantomData;

pub use display::kitty;
use itertools::Itertools;
use lender::prelude::*;

const BLOCK_SIZE: usize = 80;

#[derive(Debug)]
pub enum Node<T> {
    Leaf(Leaf<T>),
    Inner(InnerNode<T>),
}

#[derive(Debug)]
struct Leaf<T> {
    values: Vec<T>,
}

#[derive(Debug)]
struct InnerNode<T> {
    children: Vec<(Node<T>, usize)>,
}

#[derive(Debug)]
pub struct RRBTree<T> {
    root: Node<T>,
    levels: u32,
    pub len: usize,
}

impl<T: std::fmt::Debug> Extend<T> for RRBTree<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let leaf_size = max_degree_at_level::<T>(0);
        for chunk in iter.into_iter().chunks(leaf_size).into_iter() {
            let v = chunk.collect::<Vec<_>>();
            let size = v.len();
            let t = RRBTree {
                root: Node::Leaf(Leaf { values: v }),
                levels: 1,
                len: size,
            };
            self.fuse_with(t);
        }
    }
}

impl<T: std::fmt::Debug> FromIterator<T> for RRBTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut t = RRBTree::new();
        t.extend(iter);
        t
    }
}

impl<T: std::fmt::Debug> RRBTree<T> {
    pub fn new() -> Self {
        RRBTree {
            root: Node::Leaf(Leaf { values: Vec::new() }),
            levels: 1,
            len: 0,
        }
    }
    pub fn check_sizes(&self) {
        self.root.check_sizes()
    }
    pub fn iter(&self) -> Iter<T> {
        let mut iter = Iter {
            stack: vec![(&self.root as *const Node<T>, 0)],
            phantom: PhantomData,
        };
        iter.explore_towards_leaf();
        iter
    }
    pub fn insert(&mut self, index: usize, new_value: T) {
        assert!(index <= self.len);
        if index == 0 {
            let mut left = std::iter::once(new_value).collect::<RRBTree<T>>();
            std::mem::swap(&mut left, self);
            self.fuse_with(left);
        } else {
            let right = self.split_at(index);
            self.push(new_value);
            self.fuse_with(right);
        }
    }

    pub fn remove(&mut self, index: usize) {
        assert!(self.len > index);
        if index + 1 == self.len {
            self.split_at(index);
            return;
        }
        let right = self.split_at(index + 1);
        if index == 0 {
            *self = right;
        } else {
            self.split_at(index);
            self.fuse_with(right);
        }
    }
    pub fn flatten(&mut self) {
        // remove nodes which single children
        let levels_removed = self.root.flatten();
        self.levels -= levels_removed;
    }
    pub fn split_at(&mut self, index: usize) -> Self {
        let left_len = index;
        let right_len = self.len - index;
        let right_root = self.root.split_at(index, self.levels - 1);
        self.len = left_len;
        let mut right_tree = RRBTree {
            root: right_root,
            levels: self.levels,
            len: right_len,
        };
        self.flatten();
        right_tree.flatten();
        right_tree
    }
    pub fn fuse_with(&mut self, mut other: Self) {
        let max_levels = self.levels.max(other.levels);
        self.root
            .fuse_recursive(&mut other.root, self.levels - 1, other.levels - 1);
        self.levels = max_levels;

        let right_size = other.root.degree();
        if right_size != 0 {
            // new level needed
            self.root.add_father();
            self.root.children_mut().unwrap().push((other.root, 0));
            self.levels += 1;
            self.root.compute_sizes(self.levels - 1);
        }
        self.len += other.len;
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        // println!("*** looking for {index}");
        if self.len <= index {
            None
        } else {
            self.root.get(index, self.levels - 1)
        }
    }
    pub fn walk(&self) -> impl Iterator<Item = &Node<T>> {
        self.root.walk()
    }
    pub fn push(&mut self, value: T) {
        self.len += 1;
        //TODO: bon exo de borrow checker
        if let Some(new_node) = self.root.push(value) {
            self.levels += 1;
            self.root.add_father();
            self.root.children_mut().unwrap().push((new_node, 0));
        }
    }
}

impl<T: std::fmt::Debug> Leaf<T> {
    fn split_at(&mut self, index: usize, level: u32) -> Self {
        let right_values = self.values.drain(index..).collect::<Vec<_>>();
        let right_leaf = Leaf {
            values: right_values,
        };
        right_leaf
    }
}

impl<T: std::fmt::Debug> InnerNode<T> {
    fn check_sizes(&self) {
        if !self
            .children
            .last()
            .map(|(_, s)| *s == 0)
            .unwrap_or_default()
        {
            let mut sum = 0;
            for (child, child_size) in &self.children {
                sum += child.iter().count();
                assert_eq!(sum, *child_size);
            }
        }
    }
    fn direction_and_skipping_for(&self, index: usize, level: u32) -> (usize, usize) {
        let direction = if self.empty_sizes() {
            index / max_tree_size_at_level::<T>(level - 1)
        } else {
            match self.children.binary_search_by_key(&index, |(_, i)| *i) {
                Ok(i) => i + 1,
                Err(i) => i,
            }
        };
        let skipping = direction
            .checked_sub(1)
            .map(|pi| {
                if self.empty_sizes() {
                    (pi + 1) * max_tree_size_at_level::<T>(level - 1)
                } else {
                    // TODO: check invariant: pi is valid index
                    self.children[pi].1
                }
            })
            .unwrap_or_default();
        (direction, skipping)
    }
    fn split_at(&mut self, index: usize, level: u32) -> Self {
        let (direction, skipping) = self.direction_and_skipping_for(index, level);
        let first_right_child = (index - skipping != 0).then(|| {
            self.children[direction]
                .0
                .split_at(index - skipping, level - 1)
        });
        let right_children = first_right_child
            .into_iter()
            .map(|c| (c, 0))
            .chain(
                self.children
                    .drain(direction + if index == skipping { 0 } else { 1 }..),
            )
            .collect::<Vec<_>>();
        let mut right_node = InnerNode {
            children: right_children,
        };
        assert!(!right_node.children.is_empty());
        self.compute_sizes(level);
        right_node.compute_sizes(level);
        right_node
    }
    fn empty_sizes(&self) -> bool {
        self.children
            .last()
            .map(|(_, s)| *s == 0)
            .unwrap_or_default()
    }
    fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (!self.empty_sizes())
            .then(|| self.children.iter().map(|(_, s)| *s))
            .into_iter()
            .flatten()
    }
    fn compute_sizes(&mut self, level: u32) {
        let mut sizes = self.children.iter().map(|(c, _)| match c {
            Node::Leaf(l) => l.values.len(),
            Node::Inner(i) => i.sizes().last().unwrap_or_else(|| {
                max_tree_size_at_level::<T>(level as u32 - 2) * i.children.len()
            }),
        });
        let max_child_size = max_tree_size_at_level::<T>(level as u32 - 1);
        if sizes.all(|s| s == max_child_size) {
            //TODO: how to detect we are on rightmost branch ?
            // we don't need sizes
            //TODO: maybe we can just put the last size at 0
            self.children.iter_mut().for_each(|(_, s)| *s = 0);
            return;
        }
        let mut sum = 0;
        for (child, size_ref) in &mut self.children {
            let child_size = match child {
                Node::Leaf(l) => l.values.len(),
                Node::Inner(i) => i.sizes().last().unwrap_or_else(|| {
                    max_tree_size_at_level::<T>(level as u32 - 2) * i.children.len()
                }),
            };
            sum += child_size;
            *size_ref = sum;
        }
    }
}

fn max_internal_degree<T>() -> usize {
    BLOCK_SIZE / std::mem::size_of::<(Node<T>, usize)>()
}

fn max_tree_size_at_level<T>(level: u32) -> usize {
    // level 0 is leaf level
    let max_leaf_size = BLOCK_SIZE / std::mem::size_of::<T>();
    let max_internal_size = max_internal_degree::<T>();
    max_internal_size.pow(level) * max_leaf_size
}

fn max_degree_at_level<T>(level: u32) -> usize {
    if level == 0 {
        BLOCK_SIZE / std::mem::size_of::<T>()
    } else {
        max_internal_degree::<T>()
    }
}

impl<T: std::fmt::Debug> Node<T> {
    pub fn check_sizes(&self) {
        match self {
            Node::Leaf(leaf) => (),
            Node::Inner(inner_node) => inner_node.check_sizes(),
        }
    }
    pub fn iter(&self) -> Iter<T> {
        let mut iter = Iter {
            stack: vec![(self as *const Node<T>, 0)],
            phantom: PhantomData,
        };
        iter.explore_towards_leaf();
        iter
    }
    pub fn is_leaf(&self) -> bool {
        match self {
            Node::Leaf(_) => true,
            Node::Inner(_) => false,
        }
    }
    pub fn flatten(&mut self) -> u32 {
        let (mut child, removed_levels) = match self {
            Node::Leaf(_) => return 0,
            Node::Inner(inner_node) => {
                if inner_node.children.len() == 1 {
                    let mut child = inner_node.children.pop().unwrap().0;
                    let removed_below = child.flatten();
                    (child, removed_below + 1)
                } else {
                    return 0;
                }
            }
        };
        std::mem::swap(self, &mut child);
        removed_levels
    }
    pub fn walk(&self) -> impl Iterator<Item = &Node<T>> {
        let mut stack = vec![self];
        std::iter::from_fn(move || {
            let node = stack.pop();
            node.inspect(|n| match n {
                Node::Leaf(_) => (),
                Node::Inner(i) => stack.extend(i.children.iter().map(|(c, _)| c)),
            });
            node
        })
    }
    fn split_at(&mut self, index: usize, level: u32) -> Self {
        match self {
            Node::Leaf(leaf) => Node::Leaf(leaf.split_at(index, level)),
            Node::Inner(inner_node) => Node::Inner(inner_node.split_at(index, level)),
        }
    }
    pub fn get(&self, index: usize, level: u32) -> Option<&T> {
        match &self {
            Node::Leaf(leaf) => leaf.values.get(index),
            Node::Inner(inner_node) => {
                let (direction, skipping) = inner_node.direction_and_skipping_for(index, level);
                inner_node
                    .children
                    .get(direction)
                    .and_then(|(child, _)| child.get(index - skipping, level - 1))
            }
        }
    }
    fn compute_sizes(&mut self, level: u32) {
        match self {
            Node::Leaf(_) => (),
            Node::Inner(i) => i.compute_sizes(level),
        }
    }

    fn values_mut(&mut self) -> Option<&mut Vec<T>> {
        let Node::Leaf(Leaf { values: v }) = self else {
            return None;
        };
        Some(v)
    }

    fn children_mut(&mut self) -> Option<&mut Vec<(Self, usize)>> {
        let Node::Inner(InnerNode { children: c, .. }) = self else {
            return None;
        };
        Some(c)
    }
    fn children(&self) -> Option<&Vec<(Self, usize)>> {
        let Node::Inner(InnerNode { children: c, .. }) = self else {
            return None;
        };
        Some(c)
    }

    fn first_child(&mut self) -> Option<&mut (Self, usize)> {
        self.children_mut().and_then(|c| c.first_mut())
    }

    fn last_child(&mut self) -> Option<&mut (Self, usize)> {
        self.children_mut().and_then(|c| c.last_mut())
    }

    fn add_father(&mut self) {
        let mut new_node = Node::Inner(InnerNode {
            children: Vec::new(),
        });
        std::mem::swap(&mut new_node, self);
        self.children_mut().unwrap().push((new_node, 0));
    }

    fn fuse_recursive(&mut self, other: &mut Self, my_level: u32, other_level: u32) {
        if my_level > other_level {
            self.last_child()
                .unwrap()
                .0
                .fuse_recursive(other, my_level - 1, other_level);

            if other.degree() == 0 {
                self.compute_sizes(my_level);
                return;
            }
            other.add_father();
            self.pack_left(other, my_level);
        } else if other_level > my_level {
            self.fuse_recursive(
                &mut other.first_child().unwrap().0,
                my_level,
                other_level - 1,
            );
            self.add_father();
            self.fix(other, other_level);
        } else if my_level > 1 {
            self.last_child().unwrap().0.fuse_recursive(
                &mut other.first_child().unwrap().0,
                my_level - 1,
                other_level - 1,
            );
            self.fix(other, my_level);
        } else if my_level == 1 {
            // below us the leaves, let's fuse
            self.fix(other, my_level);
        } else {
            // my_level == 0
            let left = self.values_mut().unwrap();
            let right = other.values_mut().unwrap();

            let degree = left.len();
            let limit = max_degree_at_level::<T>(0);
            if degree < limit {
                let missing = (limit - degree).min(right.len());
                left.extend(right.drain(0..missing));
            }
        }
    }

    fn degree(&self) -> usize {
        match self {
            Node::Leaf(Leaf { values: v }) => v.len(),
            Node::Inner(InnerNode { children: c, .. }) => c.len(),
        }
    }

    fn fix(&mut self, other: &mut Self, level: u32) {
        match &mut self.first_child().unwrap().0 {
            Node::Leaf(_) => self.fill_children_from(other, |n| n.values_mut().unwrap(), level - 1),
            Node::Inner(_) => {
                self.fill_children_from(other, |n| n.children_mut().unwrap(), level - 1)
            }
        };
        // now, put up to P children on the left
        self.pack_left(other, level);
    }

    fn fill_children_from<C, G: Fn(&mut Self) -> &mut Vec<C>>(
        &mut self,
        other: &mut Self,
        getter: G,
        level: u32,
    ) {
        // rightmost child on the left and leftmost child on the right
        // might have a degree which is too low
        // so we need to increase their degrees.
        // how ? by moving from brothers of the leftmost child on the right.

        // first compute how much we have
        let left_size = self.last_child().unwrap().0.degree();
        let right_size: usize = other
            .children_mut()
            .unwrap()
            .iter()
            .map(|c| c.0.degree())
            .sum();
        let max_degree = max_degree_at_level::<T>(level as u32);
        let total_size = if left_size >= max_degree - 1 {
            if other.first_child().unwrap().0.degree() >= max_degree - 1 {
                return;
            }
            // if left is already ok, only rebalance the right
            right_size
        } else {
            left_size + right_size
        };

        let total_children = total_size / (max_degree - 1);
        let leftover_size = total_size - (max_degree - 1) * total_children;

        // ok we have two cases here.
        let mut target_sizes = if leftover_size > total_children {
            // there is not enough to rebalance so the last node will be of bad degree
            (std::iter::repeat(max_degree - 1)
                .take(total_children)
                .chain(std::iter::repeat(leftover_size)))
            .take(total_children + 1)
        } else {
            // there is enough
            (std::iter::repeat(max_degree)
                .take(leftover_size)
                .chain(std::iter::repeat(max_degree - 1)))
            .take(total_children)
        };

        // have a complex struct to manage all mutable borrows
        struct Children<'a, T> {
            left_child: Option<&'a mut (Node<T>, usize)>,
            right_children: &'a mut [(Node<T>, usize)],
            current_to_fill: usize,
            current_filler: usize,
        }
        impl<'a, T> Children<'a, T> {
            fn get_mut_references(&mut self) -> Option<[&mut Node<T>; 2]> {
                if let Some((left, _)) = &mut self.left_child {
                    if self.current_to_fill == 0 {
                        self.right_children
                            .get_mut(self.current_filler - 1)
                            .map(|(b, _)| [left, b])
                    } else {
                        self.right_children
                            .get_disjoint_mut([self.current_to_fill - 1, self.current_filler - 1])
                            .ok()
                            .map(|[(a, _), (b, _)]| [a, b])
                    }
                } else {
                    self.right_children
                        .get_disjoint_mut([self.current_to_fill, self.current_filler])
                        .ok()
                        .map(|[(a, _), (b, _)]| [a, b])
                }
            }
            fn advance_to_fill(&mut self) {
                self.current_to_fill += 1;
                if self.current_filler == self.current_to_fill {
                    self.current_filler += 1;
                }
            }
            fn advance_filler(&mut self) {
                self.current_filler += 1;
            }

            fn remaining_children(&mut self) -> impl Iterator<Item = &mut Node<T>> {
                let n = self.right_children.len();
                self.right_children[self.current_to_fill.min(n)..]
                    .iter_mut()
                    .map(|(c, _)| c)
            }
        }

        // there is a special case when left is already good
        // it's even possible that there is not enough to do even max_degree-1
        let mut children = Children {
            left_child: (left_size < max_degree - 1)
                .then(|| self.last_child())
                .flatten(),
            right_children: other.children_mut().unwrap().as_mut_slice(),
            current_to_fill: 0,
            current_filler: 1,
        };
        if let Some(mut target_size) = target_sizes.next() {
            while let Some([child_to_fill, filler]) = children.get_mut_references() {
                let missing = target_size.checked_sub(child_to_fill.degree());
                // if someone has more than needed then we are fine
                // since it means the "hole" is on the right
                let missing = if let Some(m) = missing { m } else { break };
                let vector_to_fill = getter(child_to_fill);
                let vector_to_drain = getter(filler);
                let (moved, advance_to_fill, advance_filler) = if missing <= vector_to_drain.len() {
                    (missing, true, vector_to_drain.len() == missing)
                } else {
                    (vector_to_drain.len(), false, true)
                };
                vector_to_fill.extend(vector_to_drain.drain(0..moved));
                if advance_to_fill {
                    // update sizes after filling
                    child_to_fill.compute_sizes(level);
                }
                if advance_filler {
                    children.advance_filler()
                }
                if advance_to_fill {
                    target_size = target_sizes.next().unwrap_or_default();
                    children.advance_to_fill()
                }
            }
        }
        for remaining_child in children.remaining_children() {
            remaining_child.compute_sizes(level);
        }
        let right_children = other.children_mut().unwrap();
        if right_children.last_mut().map(|(c, _)| c.degree()) == Some(0) {
            // remove last (now empty) child on the right
            right_children.pop();
        }
    }

    fn pack_left(&mut self, other: &mut Self, level: u32) {
        let max_degree = max_degree_at_level::<T>(level as u32);
        let missing_degree = max_degree - self.degree();
        let moving = missing_degree.min(other.degree());
        self.children_mut()
            .unwrap()
            .extend(other.children_mut().unwrap().drain(0..moving));
        self.compute_sizes(level);
        other.compute_sizes(level);
    }

    fn push(&mut self, value: T) -> Option<Node<T>> {
        match self {
            Node::Leaf(l) => {
                let max_len = max_degree_at_level::<T>(0);
                if l.values.len() < max_len {
                    l.values.push(value);
                    None
                } else {
                    Some(Node::Leaf(Leaf {
                        values: vec![value],
                    }))
                }
            }
            Node::Inner(i) => {
                if let Some(new_node) = i.children.last_mut().unwrap().0.push(value) {
                    if i.children.len() < max_internal_degree::<T>() {
                        i.children.push((new_node, 0));
                        None
                    } else {
                        Some(Node::Inner(InnerNode {
                            children: vec![(new_node, 0)],
                        }))
                    }
                } else {
                    None
                }
            }
        }
    }
}

pub struct Iter<'a, T> {
    stack: Vec<(*const Node<T>, usize)>,
    phantom: PhantomData<&'a ()>,
}

impl<'a, T: std::fmt::Debug> Iter<'a, T> {
    fn explore_towards_leaf(&mut self) {
        // invariant : anything empty is not in the stack
        loop {
            if self
                .stack
                .last()
                .map(|(node, _)| unsafe { node.as_ref().unwrap() }.is_leaf())
                .unwrap_or(true)
            {
                return;
            }
            let (node, index) = self.stack.last_mut().unwrap();
            let real_node = unsafe { node.as_ref().unwrap() };
            let children = real_node.children().unwrap();
            let child = &children[*index].0 as *const Node<T>;
            *index += 1;
            if children.len() == *index {
                self.stack.pop();
            }
            self.stack.push((child, 0));
        }
    }
}

impl<'this, 'lend, T> Lending<'lend> for Iter<'this, T> {
    type Lend = &'lend T;
}

impl<'this, T: std::fmt::Debug> Lender for Iter<'this, T> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.stack.pop().and_then(|(leaf_ptr, position)| {
            let leaf: &Leaf<T> = match unsafe { leaf_ptr.as_ref().unwrap() } {
                Node::Leaf(leaf) => leaf,
                Node::Inner(_) => unreachable!(),
            };
            if leaf.values.is_empty() {
                // special empty root case
                return None;
            }
            let value = &leaf.values[position];
            let new_position = position + 1;
            if new_position == leaf.values.len() {
                self.explore_towards_leaf()
            } else {
                self.stack.push((leaf_ptr, new_position))
            }
            Some(value)
        })
    }
}
