from functools import reduce


def robin_karp_match(pattern: str, string: str) -> int:
    N = len(pattern)
    M = len(string)
    BASE = 26
    PRIME = 1_000_000_007 # could be lower since we check for pattern, but would impact fast positive
    alpha_idx = lambda c: ord(c) - ord('a')
    custom_hash = lambda s: reduce(lambda h, c: (alpha_idx(c) + h * BASE) % PRIME, s, 0)

    pattern_h = custom_hash(pattern)
    rolling_h = custom_hash(string[:N-1])

    first_power = BASE ** (N-1)
    
    for i in range(N-1, M):
        rolling_h = (rolling_h * BASE + alpha_idx(string[i])) % PRIME
        if rolling_h == pattern_h: return i+1-N
        rolling_h = (rolling_h - alpha_idx(string[i+1-N]) * first_power) % PRIME
    return -1


def prefix(s):
    w = len(s)
    p = [0] * w
    for i in range(1, w):
        k = p[i-1]
        while k and s[i] != s[k]: k = p[k-1]
        p[i] = k + (s[i] == s[k])
    return p 

def kmp(s1, s2):
    p = prefix(s2+"#"+s1)
    offset = len(s2)+1
    return [i-offset-len(s2)+1 for i in range(offset, len(p)) if p[i] == len(s2)]



def test_prefix():
    assert prefix("ABCDE") == [0, 0, 0, 0, 0]
    assert prefix("ABABAB") == [0, 0, 1, 2, 3, 4]
    assert prefix("ABACABA") == [0, 0, 1, 0, 1, 2, 3]
    assert prefix("AAAAA") == [0, 1, 2, 3, 4]
    assert prefix("ABAABABA") == [0, 0, 1, 1, 2, 3, 2, 3]

if __name__ == "__main__":
    test_prefix()