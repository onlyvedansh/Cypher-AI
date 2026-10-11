input ciphertext


model learn from CSV file contain plain texts and cipher texts 

model extracts some features from csv file like

features = [
    "length",
    "unique_ratio",
    "repeat_ratio",
    "entropy",
    "IC",
    "vowel_ratio",
    "consonant_ratio",
    "bigram_repeat",
    "trigram_repeat"
]




length: length  of cipher text 

unique ratio : Measures how many different characters exist compared to total characters. unique characters / total characters

repeat_ratio : Measures how much characters are repeated.
For example cipher text  : KHOOR
repeat_ratio = (5 - 4) / 5
             = 0.2


entropy : Measures how random or unpredictable the characters appear.
Low entropy   → predictable/repetitive
High entropy  → more random/diverse

Example:
AAAAAAA
has very low entropy.
KQXPMZ
has higher entropy.

IC — Index of Coincidence
This measures the probability that two randomly selected characters are the same.
For example:
AAAAABBBBB
has many repeated characters, so IC is relatively high.
Something like:
KQXPMZTR
has fewer repetitions, so IC is lower.

vowel ratio: Percentage of characters that are vowels.


consonant ratio: Percentage of characters that are characters.
For:
HELLO
Consonants:
H L L
So:
consonant_ratio = 3 / 5 = 0.6

bigram repeat: A bigram means two consecutive characters.
For:
HELLO
Bigrams are:
HE
EL
LL
LO
If some bigrams repeat, we measure how much repetition exists.

trigram repeat :
Same idea, but using three characters.







Using these features , model finds out pattern and predict chances of algorithm used in cipher text



