use itertools::Itertools;

use crate::{InnerNode, Leaf, Node};

use std::io::Write;

pub fn kitty<'a, T: std::fmt::Debug + 'a, N: IntoIterator<Item = &'a Node<T>>>(
    nodes: N,
) -> std::io::Result<()> {
    {
        let mut dot_file = std::io::BufWriter::new(std::fs::File::create("test.dot")?);
        writeln!(&mut dot_file, "digraph g {{")?;
        writeln!(&mut dot_file, "node [shape = record,height=.1];")?;
        nodes.into_iter().try_for_each(|n| n.dot(&mut dot_file))?;
        writeln!(&mut dot_file, "}}")?;
    }
    std::process::Command::new("dot")
        .arg("-Tpng")
        .arg("test.dot")
        .arg("-o")
        .arg("test.png")
        .status()?;
    std::process::Command::new("kitty")
        .arg("+icat")
        .arg("test.png")
        .status()?;
    Ok(())
}

impl<T: std::fmt::Debug> InnerNode<T> {
    fn dot<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        let uid = self.uid();
        let sizes = self
            .sizes()
            .map(|s| format!("{s}"))
            .chain(std::iter::repeat("".to_owned()));
        let label = self
            .children
            .iter()
            .map(|(c, _)| c)
            .zip(sizes)
            .map(|(c, s)| format!("<c{}>{s}", c.uid()))
            .join("|");
        writeln!(&mut writer, "n{} [label=\"{}\"];", uid, label)?;
        for (child, _) in &self.children {
            let child_id = child.uid();
            writeln!(&mut writer, "\"n{}\":c{} -> n{};", uid, child_id, child_id)?;
        }
        Ok(())
    }
}

impl<T: std::fmt::Debug> Node<T> {
    pub fn dot<W: Write>(&self, writer: W) -> std::io::Result<()> {
        match self {
            Node::Leaf(l) => l.dot(writer),
            Node::Inner(i) => i.dot(writer),
        }
    }
}

impl<T: std::fmt::Debug> Leaf<T> {
    pub fn dot<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        let uid = self.uid();
        let label = self.values.iter().map(|v| format!("{:?}", v)).join("|");
        writeln!(&mut writer, "n{} [label=\"{}\"];", uid, label)?;
        Ok(())
    }
}

impl<T> Leaf<T> {
    fn uid(&self) -> usize {
        self as *const _ as usize
    }
}

impl<T> InnerNode<T> {
    fn uid(&self) -> usize {
        self as *const _ as usize
    }
}

impl<T> Node<T> {
    fn uid(&self) -> usize {
        match self {
            Node::Leaf(l) => l.uid(),
            Node::Inner(i) => i.uid(),
        }
    }
}
