#[test]
fn dump_region() {
    let code = crate::tests::MIXED_FOR_DUMP;
    let mem = memory::Memory::new(1 << 40).unwrap();
    mem.map(0x40_0000, memory::PAGE_SIZE, memory::Prot::new(true, true, true), "c").unwrap();
    mem.write_host(0x40_0000, code).unwrap();
    let (b, _) = crate::translate::region(&mem, 0x40_0000, 0x200_0000, 0x880).unwrap();
    let words: Vec<String> = b.code.iter().flat_map(|w| w.to_le_bytes()).map(|x| format!("0x{x:02x}")).collect();
    std::fs::write("/private/tmp/claude-501/-Volumes-Projects-ps5slop/4654100a-7ecb-418e-8125-b64f0f34fc90/scratchpad/asm/region.a64", words.join(" ")).unwrap();
    eprintln!("{:x?}", b.insns);
}
