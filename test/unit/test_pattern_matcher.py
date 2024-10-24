# vim: set sw=2 ts=2 colorcolumn=0:

from os import walk
from typing import Any, Pattern, Set, Dict, List, Tuple, Callable
import unittest, itertools
from tinygrad.dtype import dtypes
from tinygrad.ops import UOps, UOp, BinaryOps, TernaryOps, ReduceOps, UnaryOps # noqa: F401
from tinygrad.ops import PatternMatcher, UPat

import dfa

OTHER_TRANSITION = "OTHER"
def create_dfa(upat: UPat) -> dfa.DFA:
  accept = 0
  reject = 1
  alphabet: Set[Any] = {OTHER_TRANSITION}
  def gather_alphabet(u: UPat):
    if u.op: alphabet.update(u.op)
    if u.arg: alphabet.add(u.arg)
    if u.dtype: alphabet.update(u.dtype)
    if u.src:
      for comb in u.src:
        for v in ((next(comb),) if isinstance(comb, itertools.repeat) else comb):
          gather_alphabet(v)
  gather_alphabet(upat)
  e: Dict = dict()
  for x in alphabet: e[(reject, x)] = e[(accept, x)] = reject
  cnt = 2
  def alloc() -> int: nonlocal cnt; c = cnt; cnt += 1; return c
  def gen(u: UPat, cb: Callable[[int], None]) -> Tuple[int, Callable[[int], None]]:
    start, after_op, after_dtype = alloc(), alloc(), alloc()
    cb(start)
    accepted = set(u.op) if u.op else alphabet # can switch alphabet-> all_ops, same below
    for a in alphabet: e[(start, a)] = after_op if a in accepted else reject
    accepted = set(u.dtype) if u.dtype else alphabet 
    for a in alphabet: e[(after_op, a)] = after_dtype if a in accepted else reject
    def connect(x: int):
      accepted = {u.arg} if u.arg else alphabet
      for a in alphabet: e[(after_dtype, a)] = x if a in accepted else reject
    assert u.src # todo handle None
    assert len(u.src) == 1 # todo unfold multi-src patterns
    assert not isinstance(u.src[0], itertools.repeat) # todo handle repeats
    curcb = connect
    for i, v in enumerate(u.src[0]):
      v_start, curcb = gen(v, curcb)
      if i == 0: 
        accepted = {u.arg} if u.arg else alphabet
        for a in alphabet: e[(after_dtype, a)] = v_start if a in accepted else reject
    return start, curcb

  root, callback = gen(upat, lambda _: None)
  callback(accept)

  return dfa.DFA(
      start=root,
      inputs=alphabet,
      label=lambda x: 1 if x == accept else 0,
      transition=lambda s, c: e[(s, c)],
      outputs={0, 1}
  )

def dfa_union(dfas: List[dfa.DFA]) -> dfa.DFA:
  alphabet = set()
  for x in dfas:
    assert x.inputs
    alphabet.update(x.inputs)
  e = dict()
  state_id = dict()
  cnt = 0
  labels = dict()
  def dfs(state: Tuple[int, ...]):
    nonlocal cnt
    if (id := state_id.setdefault(state, cnt)) != cnt: return id
    labels[id] = tuple(i for i, (x, d) in enumerate(zip(state, dfas)) if d._label(x) == 1)
    cnt += 1
    for c in alphabet:
      next_state = tuple(d._transition(s, c if c in d.inputs else OTHER_TRANSITION) for s, d in zip(state, dfas))
      e[(state_id[state], c)] = dfs(next_state)
    return id

  start = dfs(tuple(int(d.start) for d in dfas))
  return dfa.DFA(
      start=start,
      inputs=alphabet,
      label=lambda x: labels[x],
      transition=lambda s, c: e[(s, c)],
      outputs=labels.values()
  )


def create_multi_dfa(upats: List[UPat]) -> dfa.DFA:
  return dfa_union([create_dfa(u) for u in upats])

def get_dfa_word(dfa: dfa.DFA, u: UOp) -> List:
  assert dfa.inputs
  return [x if x in dfa.inputs else OTHER_TRANSITION for x in (u.op, u.dtype, u.arg)] +\
      list(itertools.chain.from_iterable(map(lambda x: get_dfa_word(dfa, x), u.src)))

class TestPatternMatcher(unittest.TestCase):
  def test_dfa(self):
    upats = [UPat(UOps.CONST, name="x", dtype=dtypes.float, src=()), UPat(UOps.CONST, name="x", dtype=dtypes.float, src=())];
    dfa = create_multi_dfa(upats)
    assert dfa.label(get_dfa_word(dfa, UOp(UOps.CONST, dtypes.float, arg=1.0))) == (0,1)
    assert dfa.label(get_dfa_word(dfa, UOp(UOps.CONST, dtypes.int, arg=1.0))) == ()

  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  @unittest.skip("closures aren't supported on pattern matchers")
  def test_match_sz_0(self):
    match_cnt = 0
    def fxn(x):
      nonlocal match_cnt
      match_cnt += 1
      assert len(x.src) == 0
      return UOp(UOps.CONST, src=(UOp(UOps.CONST),))
    matcher = PatternMatcher([(UPat(UOps.CONST, src=(), name="x"), fxn)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    # second rewrite shouldn't match anything
    c1 = matcher.rewrite(c1)
    c1 = matcher.rewrite(c1)
    self.assertEqual(match_cnt, 1)

  def test_match_sz_0_ctx(self):
    def fxn(ctx, x):
      ctx.append(True)
      assert len(x.src) == 0
      return UOp(UOps.CONST, src=(UOp(UOps.CONST),))
    matcher = PatternMatcher([(UPat(UOps.CONST, src=(), name="x"), fxn)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    # second rewrite shouldn't match anything
    ctx = []
    c1 = matcher.rewrite(c1, ctx)
    c1 = matcher.rewrite(c1, ctx)
    self.assertEqual(len(ctx), 1)

  def test_uop(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1, c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([(UPat({UOps.CONST, UOps.CAST}, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c2 = UOp(UOps.CAST, dtypes.int, (c1,))
    c3 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c4 = UOp(UOps.ALU, dtypes.float, (c3, c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_arg(self):
    matcher = PatternMatcher([
      (UPat(UOps.CONST, arg=0, name="x"), lambda x: x),
      (UPat(UOps.CONST, arg=False, name="x"), lambda x: x),
      (UPat(UOps.ALU, arg=BinaryOps.MAX, name="x"), lambda x: x),
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c2 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MAX)
    c4 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MUL)
    c5 = UOp(UOps.CONST, dtypes.int, arg=-1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_filter_arg(self):
    matcher = PatternMatcher([
      (UPat(UOps.ALU, arg=BinaryOps.MUL, src=[UPat(UOps.CONST, name="c"), UPat(UOps.CONST, arg=2)], name="x"),
       lambda x,c: x if c.arg in {1, -1} else None)
    ])
    y1 = UOp(UOps.CONST, dtypes.int, arg=1)
    y2 = UOp(UOps.CONST, dtypes.int, arg=2)
    y3 = UOp(UOps.CONST, dtypes.int, arg=-1)
    c1 = UOp(UOps.ALU, dtypes.int, (y1, y2), BinaryOps.MUL)
    c2 = UOp(UOps.ALU, dtypes.int, (y2, y2), BinaryOps.MUL)
    c3 = UOp(UOps.ALU, dtypes.int, (y3, y2), BinaryOps.MUL)
    c4 = UOp(UOps.ALU, dtypes.int, (y2, y1), BinaryOps.MUL)
    c5 = UOp(UOps.ALU, dtypes.int, (y2, y3), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)

  def test_dup_name(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST, name="y"), UPat(UOps.CONST, name="y"))), lambda x, y: x)])
    y1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    y2 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c1 = UOp(UOps.ALU, dtypes.float, (y1, y1), BinaryOps.ADD)
    c2 = UOp(UOps.ALU, dtypes.float, (y1, y2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c1)

  def test_dtype(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float32), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype={dtypes.float32, dtypes.float64}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(UOps.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_src_one(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.CONST))), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.ALU))), lambda x: x)])
    c4 = UOp(UOps.ALU, dtypes.float, (c1,c3), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c3,c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_src_permutations(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=[UPat(UOps.CONST), UPat(UOps.ALU)]), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c3,c2), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c3,c4), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)

  def test_src_repeat(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=UPat(UOps.CONST)), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_allow_len(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST),), allow_any_len=True, arg=TernaryOps.MULACC), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.CONST, dtypes.float, arg=3.0)
    c4 = UOp(UOps.ALU, dtypes.float, (c1,), UnaryOps.EXP2)
    c5 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c1,c2,c3), TernaryOps.MULACC)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)
    self.assertEqual(matcher.rewrite(c6), c6)

  def test_deep_src_permutations(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    u1 = (c1 + c2) + c1
    u2 = (c2 + c1) + c1
    matcher = PatternMatcher([
      (UPat(UOps.ALU, src=[UPat(UOps.ALU, src=[UPat(name='a'), UPat(name='b')]), UPat(name='b')]), lambda a,b: b)
    ])
    self.assertIsNotNone(matcher.rewrite(u1))
    self.assertIsNotNone(matcher.rewrite(u2))

  def _assert_eq_upat(self, a:UPat, b:UPat):
    assert (sorted(map(str,a.op)) if a.op else [] == (sorted(map(str,b.op)) if b.op else []))
    assert (sorted(a.dtype) if a.dtype else [] == (sorted(b.dtype) if b.dtype else []))
    assert (a.name, type(a.src)) == (b.name, type(b.src))
    def simple_src(u:UPat):
      if u.src is None: return []
      if isinstance(u.src, itertools.repeat): return next(u.src[0])
      return u.src[0]
    for a,b in zip(simple_src(a), simple_src(b)): self._assert_eq_upat(a, b)

if __name__ == '__main__':
  unittest.main(verbosity=2)
