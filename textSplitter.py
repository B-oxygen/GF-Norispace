#Let's read a text file
filepath = "../../Sessions_Part2\datasets\Harry Potter 1 - Sorcerer_s Stone.txt"
with open(filepath,'r') as f:
    hp_book = f.read()
    

from langchain.text_splitter import CharacterTextSplitter

def len_func(text):
    return len(text)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size = 1200,
    chunk_overlap = 100,
    length_function = len_func,
    is_separator_regex= False)


para_list = text_splitter.create_documents(texts = [hp_book])
para_list[:2]

Document(page_content="Harry Potter and the Sorcerer's Stone\n\n\nCHAPTER ONE\n\nTHE BOY WHO LIVED\n\nMr. and Mrs. Dursley, of number four, Privet Drive, were proud to say\nthat they were perfectly normal, thank you very much. They were the last\npeople you'd expect to be involved in anything strange or mysterious,\nbecause they just didn't hold with such nonsense.\n\nMr. Dursley was the director of a firm called Grunnings, which made\ndrills. He was a big, beefy man with hardly any neck, although he did\nhave a very large mustache. Mrs. Dursley was thin and blonde and had\nnearly twice the usual amount of neck, which came in very useful as she\nspent so much of her time craning over garden fences, spying on the\nneighbors. The Dursleys had a small son called Dudley and in their\nopinion there was no finer boy anywhere."),
 Document(page_content="The Dursleys had everything they wanted, but they also had a secret, and\ntheir greatest fear was that somebody would discover it. They didn't\nthink they could bear it if anyone found out about the Potters. Mrs.\nPotter was Mrs. Dursley's sister, but they hadn't met for several years;\nin fact, Mrs. Dursley pretended she didn't have a sister, because her\nsister and her good-for-nothing husband were as unDursleyish as it was\npossible to be. The Dursleys shuddered to think what the neighbors would\nsay if the Potters arrived in the street. The Dursleys knew that the\nPotters had a small son, too, but they had never even seen him. This boy\nwas another good reason for keeping the Potters away; they didn't want\nDudley mixing with a child like that.\n\nWhen Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story\nstarts, there was nothing about the cloudy sky outside to suggest that\nstrange and mysterious things would soon be happening all over the\ncountry. Mr. Dursley hummed as he picked out his most boring tie for\nwork, and Mrs. Dursley gossiped away happily as she wrestled a screaming\nDudley into his high chair.\n\nNone of them noticed a large, tawny owl flutter past the window.")]
 

first_chunk = para_list[0]

first_chunk.metadata = {"source":filepath}
first_chunk.metadata

{'source': '../../Sessions_Part2\\datasets\\Harry Potter 1 - Sorcerer_s Stone.txt'}

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n", " "],
    chunk_size = 200,
    chunk_overlap = 100,
    length_function = len_func,
    is_separator_regex=False
)

chunk_list = text_splitter.create_documents(texts = [hp_book]