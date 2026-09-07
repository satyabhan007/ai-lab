"""
tokenizer/step1_why_tokenize.py
===============================
LESSON 1: WHY do we need a tokenizer at all?

Neural networks can only do math on NUMBERS.
You type "hello" - the network must receive numbers like [104, 101, 108, 108, 111].

This lesson shows the journey: TEXT -> BYTES -> NUMBERS,
and why raw bytes alone are not good enough (too long, no meaning).
"""

print("=" * 60)
print("STEP 1: Why tokenize? Text must become numbers")
print("=" * 60)

text = "hello"
print(f"\nOur text: {text!r}")

# UTF-8: the universal way to turn ANY text into bytes
raw = text.encode('utf-8')
print(f"As UTF-8 bytes: {list(raw)}")
print(f"  ('h'={ord('h')}, 'e'={ord('e')}, 'l'={ord('l')}, 'l'={ord('l')}, 'o'={ord('o')})")

# every character the computer knows is secretly a number
print("\nEvery character is secretly a number (its code point):")
for ch in "abcXYZ019 !?":
    print(f"  {ch!r} -> {ord(ch)}")

# what about emoji / other languages? UTF-8 uses MULTIPLE bytes
emoji = "\U0001F642"   # a smiley face
raw_emoji = emoji.encode('utf-8')
print(f"\nAn emoji {emoji!r} is {len(raw_emoji)} bytes: {list(raw_emoji)}")
print("  -> byte-level tokenization works for EVERY language on Earth,")
print("     because ANY text can be written as bytes. No crashes, no 'unknown word'.")

print("\n" + "=" * 60)
print("THE PROBLEM: raw bytes are SAFE but DUMB")
print("=" * 60)

sentence = "the cat sat on the mat"
ids = list(sentence.encode('utf-8'))
print(f"\n{sentence!r}")
print(f"  = {len(ids)} separate numbers: {ids[:12]} ...")
print("\n  Two problems:")
print("   1. TOO LONG: 'the' alone is 3 numbers. A whole book = millions of them.")
print("   2. NO MEANING: the network sees 116,104,101 - it has no idea that")
print("      t+h+e together form the most common word in English: 'the'.")

print("\n  THE FIX (BPE): glue common byte pairs into single tokens.")
print("     't' + 'h'  ->  'th'   (one token)")
print("     'th' + 'e' ->  'the'  (one token!)")
print("  -> shorter sequences AND tokens that carry meaning.")
print("  That is exactly what step2 will train from scratch.")