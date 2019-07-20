'''和拉丁语系不同，亚洲语言是不用空格分开每个有意义的词的。而当我们进行自然语言处理的时候，大部分情况下，词汇是我们对句子和文章理解的基础，
   因此需要一个工具去把完整的文本中分解成粒度更细的词。jieba就是这样一个非常好用的中文工具，是以分词起家的，但是功能比分词要强大很多。'''
'''基本分词函数与用法:
      jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)
      jieba.cut 方法接受三个输入参数:
        需要分词的字符串
        cut_all 参数用来控制是否采用全模式
        HMM 参数用来控制是否使用 HMM 模型
      jieba.cut_for_search 方法接受两个参数:
        需要分词的字符串
        是否使用 HMM 模型。
    该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细'''
import jieba

seg_list = jieba.cut("我在学习自然语言处理", cut_all=True)
print seg_list
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我在学习自然语言处理", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他毕业于上海交通大学，在百度深度学习研究院进行研究")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

# jieba.lcut以及jieba.lcut_for_search直接返回 list
result_lcut = jieba.lcut("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")
print result_lcut
print " ".join(result_lcut)
print " ".join(jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造"))

'''添加用户自定义词典
      很多时候我们需要针对自己的场景进行分词，会有一些领域内的专有词汇。
      1.可以用jieba.load_userdict(file_name)加载用户字典
      2.少量的词汇可以自己用下面方法手动添加：
          用 add_word(word, freq=None, tag=None) 和 del_word(word) 在程序中动态修改词典
          用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。'''
print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))
print(jieba.suggest_freq(('中', '将'), True))
print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))

'''关键词提取
      基于 TF-IDF 算法的关键词抽取
      import jieba.analyse
      jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
        sentence 为待提取的文本
        topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
        withWeight 为是否一并返回关键词权重值，默认值为 False
        allowPOS 仅包括指定词性的词，默认值为空，即不筛选'''
import jieba.analyse as analyse
lines = open('NBA.txt').read()
print "  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=()))

lines = open(u'西游记.txt').read()
print "  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=()))

'''词性标注
      jieba.posseg.POSTokenizer(tokenizer=None) 新建自定义分词器，tokenizer 参数可指定内部使用的 jieba.Tokenizer 分词器。
      jieba.posseg.dt 为默认词性标注分词器。
      标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法。'''
import jieba.posseg as pseg
words = pseg.cut("我爱自然语言处理")
for word, flag in words:
    print('%s %s' % (word, flag))
    
'''并行分词
      原理：将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，然后归并结果，从而获得分词速度的可观提升 基于 python 自带的 multiprocessing 模块，
      目前暂不支持 Windows.
      用法：
        jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
        jieba.disable_parallel() # 关闭并行分词模式'''
import sys
import time
import jieba

jieba.enable_parallel()
content = open(u'西游记.txt',"r").read()
t1 = time.time()
words = "/ ".join(jieba.cut(content))
t2 = time.time()
tm_cost = t2-t1
print('并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

jieba.disable_parallel()
content = open(u'西游记.txt',"r").read()
t1 = time.time()
words = "/ ".join(jieba.cut(content))
t2 = time.time()
tm_cost = t2-t1
print('非并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

# Tokenize：返回词语在原文的起止位置.注意，输入参数只接受 unicode
print "这是默认模式的tokenize"
result = jieba.tokenize(u'自然语言处理非常有用')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

print "\n-----------我是神奇的分割线------------\n"

print "这是搜索模式的tokenize"
result = jieba.tokenize(u'自然语言处理非常有用', mode='search')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

# ChineseAnalyzer for Whoosh 搜索引擎
from __future__ import unicode_literals
import sys,os
sys.path.append("../")
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser

analyzer = jieba.analyse.ChineseAnalyzer()
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
    
if not os.path.exists("tmp"):
    os.mkdir("tmp")

ix = create_in("tmp", schema) # for create new index
#ix = open_dir("tmp") # for read only
writer = ix.writer()

writer.add_document(
    title="document1",
    path="/a",
    content="This is the first document we’ve added!"
)

writer.add_document(
    title="document2",
    path="/b",
    content="The second one 你 中文测试中文 is even more interesting! 吃水果"
)

writer.add_document(
    title="document3",
    path="/c",
    content="买水果然后来世博园。"
)

writer.add_document(
    title="document4",
    path="/c",
    content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
)

writer.add_document(
    title="document4",
    path="/c",
    content="咱俩交换一下吧。"
)

writer.commit()
searcher = ix.searcher()
parser = QueryParser("content", schema=ix.schema)

for keyword in ("水果世博园","你","first","中文","交换机","交换"):
    print(keyword+"的结果为如下：")
    q = parser.parse(keyword)
    results = searcher.search(q)
    for hit in results:
        print(hit.highlights("content"))
    print("\n--------------我是神奇的分割线--------------\n")

for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
    print(t.text)


      

         
