use lender::Lender;
use rrb::{kitty, RRBTree};

fn main() -> std::io::Result<()> {
    //    let mut t: RRBTree<u64> = RRBTree::new();
    //    t.check_sizes();
    //    for _ in 0..128 {
    //        println!("***************");
    //        t.extend(0..12);
    //        println!("{}", t.len);
    //        kitty(t.walk()).unwrap();
    //        t.check_sizes();
    //    }
    //    let taille = t.iter().count();
    //    assert_eq!(taille, t.len);
    //    println!("15488: {:?}", t.iter().nth(15488));
    //    let s = t.get(15488);
    //    println!("s {s:?}");
    let n = 30 + rand::random::<u32>() % 40 + 5;
    let split = rand::random::<u32>() % n;
    println!("n : {n} split {split}");
    let mut r1: RRBTree<u32> = (0..split).collect();
    println!("r1");
    kitty(r1.walk())?;
    let r2: RRBTree<u32> = (split..n).collect();
    println!("r2");
    kitty(r2.walk())?;
    println!("concatenation");
    r1.fuse_with(r2);
    kitty(r1.walk()).unwrap();
    for x in 0..n {
        assert_eq!(r1.get(x as usize), Some(&x));
    }
    r1.remove(10);
    println!("after removal");
    kitty(r1.walk()).unwrap();
    println!("let's try to pop");
    r1.remove(33);
    println!("after removal");
    kitty(r1.walk()).unwrap();

    // r1.iter().for_each(|v| println!("{v}"));
    Ok(())
}
