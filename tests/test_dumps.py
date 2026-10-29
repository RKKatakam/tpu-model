from parser import dumps

TEXT = """= control target key start
LH: loop header
= control target key end

     0   :  { %6 = vsyncpa [#allocation4], 0 /* loc("a.py":1:2) */  ;;  %s13_s8 = sshll.u32 %s115_s0, 4 }
   0x3   :  { %18 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s14_s8 /*
base_bounds: (1, 1)
strides: (1, 1) */ }
   0x4   :  { %93 = dma.done.wait [#allocation4], 128 /* pipeline-emitter-dma-wait */ }
"""


def test_bundle_lines_ignores_comments_and_header():
    lines = dumps.bundle_lines(TEXT)
    assert lines == [
        "0 : { %6 = vsyncpa [#allocation4], 0 ;; %s13_s8 = sshll.u32 %s115_s0, 4 }",
        "0x3 : { %18 = dma.hbm_to_vmem [thread:$0] %s14_s8 }",
        "0x4 : { %93 = dma.done.wait [#allocation4], 128 }",
    ]


def test_annotation_only_change_compares_equal():
    other = TEXT.replace('loc("a.py":1:2)', 'loc("b.py":9:9)')
    assert dumps.bundle_lines(other) == dumps.bundle_lines(TEXT)
