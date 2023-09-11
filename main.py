import nltk
import requests
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from urllib.request import urlopen
from bs4 import BeautifulSoup
import math
import tkinter as tk
import tkinter.ttk as ttk
from rank_bm25 import BM25Okapi
import numpy as np
import webbrowser


def fetch_content(url):
    response = requests.get(url)
    content = response.text
  
def open_url(event):
    url = event.widget.cget("text")
    webbrowser.open(url)


class PageRank:
    def __init__(self, nums, graph):
        self.total_nodes = nums
        self.graph = graph
        self.page_rank = np.full(self.total_nodes, 0.2)

    def calc_outgoing_links(self, node_number):
        outgoing_links = 0
        for i in range(self.total_nodes):
            if self.graph[node_number][i] == 1:
                outgoing_links += 1

        return outgoing_links

    def calc_page_rank_for_node(self, node_number):
        new_page_rank = 0
        for external_link in range(self.total_nodes):
            if self.graph[external_link][node_number]:
                outgoing_links = self.calc_outgoing_links(external_link)
                new_page_rank += self.page_rank[external_link] / outgoing_links
        return new_page_rank

    def calc_all_pages(self):
        for i in range(10):
            new_page_ranks = np.zeros(self.total_nodes)
            for page in range(self.total_nodes):
                new_page_ranks[page] = self.calc_page_rank_for_node(page)
            self.page_rank = new_page_ranks
        return new_page_ranks


root = tk.Tk()
ws = root.winfo_screenwidth()
wh = root.winfo_screenheight()
w = 700
h = 500
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
root.geometry("%dx%d+%d+%d" % (w, h, x, y))
root.title("Findfinity")
root.iconbitmap(r"Images\ico.ico")
root.tk_setPalette(background="#6b6b6b",foreground="#fdfcff",activeBackground="#110000",activeForeground="#6b6b6b")
frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=10, pady=10)


class Crawler:
    def __init__(self):
        self.links_to_visit = []
        self.visited_links = []
        self.max_visited_links = 20
        self.invindex = {}
        self.tfidf_inv_index = {}
        self.bm25 = BM25Okapi
        self.links = []
        self.link_links = []
        self.page_inv_index = {}

    def start(self):
        startLabel = tk.Label(frame, text="Crawler:", font=("Arial", 12))
        startLabel.grid(row=0, column=0, sticky="w")
        startEntry = tk.Entry(frame,width=60,font=("Arial", 13),bg="white",relief='flat',fg="black",insertbackground="black")
        startEntry.grid(row=1, column=0, padx=(0, 10))
        depthEntry = tk.Entry(frame,width=5,font=("Arial", 13),relief='flat',bg="white",fg="black",insertbackground="black")
        depthEntry.grid(row=1, column=1, padx=(0, 10))
        startButton = tk.Button(frame,width=5,relief='flat', text="Start",bg="#0c6098",command=lambda: self.get_data(startEntry, depthEntry),font=("Arial", 12))
        startButton.grid(row=1, column=2, padx=(0, 10)) 
        searchLabel = tk.Label(frame, text="Search:", font=("Arial", 12))
        searchLabel.grid(row=7, column=0, sticky="w")
        searchEntry = tk.Entry(frame,width=67,relief='flat', font=("Arial", 13), bg="white", fg="black",insertbackground="black")
        searchEntry.grid(row=8, column=0, columnspan=2)
        ranks = ["tf-idf", "page-rank", "BM25"]
        var = tk.StringVar(None, ranks[0])
        radio_frame = tk.Frame(frame)
        for i in range(len(ranks)):
            r = tk.Radiobutton(radio_frame,text=ranks[i],variable=var,value=ranks[i],selectcolor="#6b6b6b",font=("Arial", 13))
            r.grid(row=0, column=i)  
            radio_frame.grid(row=4, column=0, columnspan=2, sticky="n")
        searchButton = tk.Button(frame,relief='flat',width=5,text="Search",bg="#0c6098",command=lambda: self.searchwindow(table_frame, searchEntry, var.get()),font=("Arial", 12))
        startyText = tk.Text(frame,height=8,width=50,wrap="word",highlightthickness=0,borderwidth=0)   
        startyText.grid(row=3, column=0, columnspan=3, padx=0, pady=(10, 0), sticky="ew")
        searchButton.grid(row=8, column=2)  
        table_frame = tk.Frame(root)
        table_frame.configure(height=150)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        root.mainloop()

    def lemma(self, tokens):
        lemlist = []
        ll = WordNetLemmatizer()
        n = pos_tag(tokens)
        for word, tag in n:
            wt = tag[0].lower()
            wt = wt if wt in ["a", "r", "n", "v"] else None
            if not wt:
                lem = word
            else:
                lem = ll.lemmatize(word, wt)
            lemlist.append(lem)
        return lemlist

    def create_graph(self):
        graph = np.zeros((len(self.links), len(self.links)))
        for i in range(len(self.links)):
            for l in range(len(self.links)):
                if (
                    self.links[l] != self.links[i]
                    and self.links[l] in self.link_links[i]
                ):
                    graph[i, l] = 1
        p1 = PageRank(len(self.links), graph)
        rankss = p1.calc_all_pages()
        print(rankss)

        for word in self.invindex.keys():
            new_order = []
            links = self.invindex.get(word)
            ranks = []
            for link in links:
                rank = rankss[self.links.index(link)]
                ranks.append(rank)
            ranks = sorted(ranks, reverse=True)
            for r in ranks:
                for j in links:
                    if rankss[self.links.index(j)] == r and j not in new_order:
                        new_order.append(j)
                        break
            self.page_inv_index.update({word: new_order})
            
    def get_link_content(self, link):
        content = requests.get(link).content
        return content

    def create_inv_index(self, t):
        ttf = {}
        tbm = []
        for i in t.keys():
            terms = t.get(i)
            lt = self.lemma(terms)
            ttf.update({i: lt})
            tbm.append(lt)
            for k in range(len(terms)):
                terms[k] = terms[k].lower()
                lt[k] = lt[k].lower()
                if lt[k] not in stopwords.words("english") and not lt[k].isdigit():
                    if lt[k] in self.invindex.keys():
                        f = self.invindex.get(lt[k])
                    else:
                        f = []
                    if i not in f:
                        f.append(i)
                        self.invindex.update({lt[k]: f})
        self.create_tfidf_ranking(ttf)
        self.create_graph()
        self.bm25 = BM25Okapi(tbm)
        return

    def search(self, query, var):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        x = tokenizer.tokenize(query)
        y = self.lemma(x)
        bindex = []
        results = []
        for i in range(len(x)):
            bindex.append(self.find_similarity(y[i], var))
        for i in bindex[0]:
            bi = True
            for j in range(1, len(bindex)):
                if i not in bindex[j]:
                    bi = False
                    break
            if bi:
                results.append(i)
        return results

    def create_bm25_ranking(self, word, array):
        doc_scores = self.bm25.get_scores(word)
        ranks = []
        rr = {}
        new_order = []
        for i in array:
            rank = doc_scores[self.links.index(i)]
            ranks.append(rank)
            rr.update({i: rank})
        ranks = sorted(ranks, reverse=True)
        for r in ranks:
            for j in rr.keys():
                if rr.get(j) == r:
                    new_order.append(j)
                    break
        return new_order

    def find_similarity(self, word, var):
        if var != "BM25":
            index = {}
            if var == "tf-idf":
                index = self.tfidf_inv_index
            elif var == "page-rank":
                index = self.page_inv_index
            similar = index.get(word.lower())
            for i in index.keys():
                try:
                    if (wordnet.synsets(word.lower())[0].wup_similarity(wordnet.synsets(i)[0])>= 0.7 and i != word.lower()):
                        for x in index.get(i):
                            if x not in similar:
                                similar.append(x)
                except:
                    None
        else:
            similar = self.create_bm25_ranking(word.lower(), self.invindex.get(word.lower()))
            for i in self.invindex.keys():
                try:
                    if (wordnet.synsets(word.lower())[0].wup_similarity(wordnet.synsets(i)[0])  >= 0.7and i != word.lower()):
                        for x in self.create_bm25_ranking(i, self.invindex.get(i)):
                            if x not in similar:
                                similar.append(x)
                except:
                    None
        return similar

    def get_text(self, url):
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract() 
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        return tokenizer.tokenize(text)

    def scan_links(self, content, x):
        links = re.findall('<a href="([^"]+)"', str(content))
        ll = []
        pat = re.compile("http?")
        media = ["facebook","instagram","twitter","account","youtube","register","login","wordpress","subscribe"]
        for link in links:
            x = True
            for m in media:
                if m in link:
                    x = False
                    break
            if pat.match(link) and x:
                self.links_to_visit.append(link)
                ll.append(link)
        if x:
            self.link_links.append(ll)
        return

    def crawl(self, main_url, depth):
        self.max_visited_links = depth
        self.links_to_visit.append(main_url)
        terms = {}
        while len(self.visited_links) < self.max_visited_links:
            link = self.links_to_visit.pop()
            while link in self.visited_links:
                link = self.links_to_visit.pop()
            content = self.get_link_content(link)
            x = False
            try:
                t = self.get_text(link)
                x = True
                self.links.append(link)
                terms.update({link: t})
            except:
                None
            self.scan_links(content, x)
            self.visited_links.append(link)
            startyText = tk.Text(frame, height=8,width=50,wrap="word",highlightthickness=0,borderwidth=0)
            startyText.grid(row=3,column=0,columnspan=3,padx=0,pady=(10, 0),sticky="ew")
            startyText.insert(tk.END, "Visited URL: " + link + "\n")
            startyText.insert(tk.END, "Fetching URL: " + ", ".join(self.visited_links) + "\n")
        self.create_inv_index(terms)
        return

    def computeTF(self, d, ttf):
        raw_tf = dict.fromkeys(self.invindex.keys(), 0)
        norm_tf = {}
        doc = ttf.get(d)
        bow = len(doc)
        for word in doc:
            if word in self.invindex.keys():
                raw_tf[word] += 1
        for word, count in raw_tf.items():
            norm_tf[word] = count / float(bow)
        return raw_tf, norm_tf

    def computeIdf(self, doclist):
        idf = {}
        idf = dict.fromkeys(doclist[0].keys(), float(0))
        for doc in doclist:
            for word, val in doc.items():
                if val > 0:
                    idf[word] += 1
        for word, val in idf.items():
            idf[word] = math.log10(len(doclist) / float(val))
        return idf

    def computeTfidf(self, norm_tf, idf):
        tfidf = {}
        for word, val in norm_tf.items():
            tfidf[word] = val * idf[word]
        return tfidf

    def create_tfidf_ranking(self, ttf):
        tfs = []
        tfidf_ranks = {}
        for doc in ttf.keys():
            tf_dict, norm_tf_dict = self.computeTF(doc, ttf)
            tfs.append(tf_dict)
            ttf.update({doc: norm_tf_dict})
        idf = self.computeIdf(tfs)
        for doc in ttf.keys():
            tfidf = self.computeTfidf(ttf.get(doc), idf)
            tfidf_ranks.update({doc: tfidf})
        for word in self.invindex.keys():
            new_order = []
            links = self.invindex.get(word)
            ranks = []
            for link in links:
                rank = tfidf_ranks.get(link).get(word)
                ranks.append(rank)
            ranks = sorted(ranks, reverse=True)
            for r in ranks:
                for j in links:
                    if tfidf_ranks.get(j).get(word) == r and tfidf_ranks.get(j).get(
                        word
                    ):
                        new_order.append(j)
                        break
            self.tfidf_inv_index.update({word: new_order})
        return

    def get_data(self, text, depth):
        url = text.get()
        pat = re.compile("http?")
        if url and pat.match(url):
            x = depth.get()
            if x.isdigit():
                x = int(x)
                text.delete(0, tk.END)
                depth.delete(0, tk.END)
                self.crawl(url, x)
                tk.messagebox.showinfo("Findfinity", "All Up")
            else:
                tk.messagebox.showerror("Findfinity Error:", "Depth is Required")
        else:
            tk.messagebox.showerror("Findfinity Error:", "Bad URL")
        return

    def searchwindow(self, table_frame, text, var):
        textbox = tk.Text(table_frame, height=10, width=79, wrap="word")
        query = text.get()
        if query:
            text.delete(0, tk.END)
            results = self.search(query, var)
            if textbox.get("1.0", tk.END):
                textbox.delete("1.0", tk.END)
            vertscroll = ttk.Scrollbar(table_frame)
            vertscroll.config(command=textbox.yview)
            s = tk.Label(frame, text=var + " ranked results:\n", font=("Arial", 12))
            s.grid(row=9, column=0, sticky="w")
            for i, link in enumerate(results):
                link_label = tk.Label(
                    table_frame, text=link, fg="white", cursor="hand2", anchor="w")
                link_label.grid(row=i, column=0, sticky="w")
                link_label.bind("<Button-1>", open_url)
        else:
            tk.messagebox.showerror("Findfinity Error:", "Seacrh Box Should not be Empty")
        return


c1 = Crawler()
c1.start()
