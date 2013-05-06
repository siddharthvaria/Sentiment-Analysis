#! /usr/bin/env python
import nltk
from collections import defaultdict as ddict
fpatterns_pos=open('patterns_pos.txt','w')
fpatterns_neg=open('patterns_neg.txt','w')
common_dir='E:\\Study_Material\\quarter_2\\SNLP\\project_data\\'
def main():
    dirs=['negm','posm']
    swn=senti_wordnet()
    num_of_files=101
    for directory in dirs:
        fresults=open(common_dir+directory+'\\results1.txt','w')
        fresults.write('*****'+directory+'*****'+'\n')
        polarity_counter=ddict(int)
        print '****************************************'+directory+'****************************************'
        print 'Processing Files...'
        for i in xrange(num_of_files):
            freview=open(common_dir+directory+'\\r'+str(i)+'.txt','r')
            lines = [line.strip() for line in freview.readlines()]
            freview.close()
            pol=extract_features_n_get_polarity(lines,swn,directory)
            if pol>0:
               polarity_counter['pos']+=1 
               fresults.write('review:%d\tpolarity:positive\n'%(i))
            elif pol<0:
               polarity_counter['neg']+=1  
               fresults.write('review:%d\tpolarity:negative\n'%(i))
            else: 
               fresults.write('review:%d\tpolarity:neutral\n'%(i))
            if i%10==0:
               print '. ',
        print '\n'   
        if directory=='negm':
           accuracy=(float(polarity_counter['neg'])/num_of_files)*100
        else:
           accuracy=(float(polarity_counter['pos'])/num_of_files)*100
        fresults.write('Accuracy:%f'%(accuracy))
        fresults.close()
        fresults.close()    

def extract_features_n_get_polarity(ls,swn,directory):
    patterns=[]
    for l in ls:
        l=l.replace('-',' ')
        lwrds=nltk.word_tokenize(l)
        tgdlwrds=nltk.pos_tag(lwrds)
        count_critique=tgdlwrds.count('critique :')
        count_comments=tgdlwrds.count('comments :')
        if count_critique>0 or count_comments>0 :
           patterns=[]
           continue
        tgdlwrds.insert(0,('<s>','START'))
        tgdlwrds.append(('</s>','END'))
        tgdlwrds_trigrams=nltk.trigrams(tgdlwrds)
        for tg in tgdlwrds_trigrams:
            if tg[0][1]=='JJ' and (tg[1][1]=='NN' or tg[2][1]=='NNS'):
                patterns.append(tg[0][0].lower()+'#'+'a')
            if ((tg[0][1]=='RB' or tg[0][1]=='RBR' or tg[0][1]=='RBS') and tg[1][1]=='JJ' and not(tg[2][1]=='NN' or tg[2][1]=='NNS')):
                patterns.append(tg[1][0].lower()+'#'+'a')
            if (tg[0][1]=='JJ' and tg[1][1]=='JJ' and not(tg[2][1]=='NN' or tg[2][1]=='NNS')):
                patterns.append(tg[0][0].lower()+'#'+'a')
                patterns.append(tg[1][0].lower()+'#'+'a')
            if ((tg[0][1]=='NN' or tg[0][1]=='NNS') and tg[1][1]=='JJ' and not(tg[2][1]=='NN' or tg[2][1]=='NNS')):
                patterns.append(tg[1][0].lower()+'#'+'a')
            if ((tg[0][1]=='RB' or tg[0][1]=='RBR' or tg[0][1]=='RBS') and not(tg[1][1]=='VB' or tg[1][1]=='VBG' or tg[1][1]=='VBN' or tg[1][1]=='VBD')):
                patterns.append(tg[0][0].lower()+'#'+'r')
    polarity=0
    ncount_obj=0
    ncount_sub=0
    ncount_notf=0
    for p in patterns:
        score=swn.get_score2(p)
        if score==-1:
           ncount_notf+=1
           if directory=='pos':
               fpatterns_pos.write(p+'\n')
           else:
               fpatterns_neg.write(p+'\n')
           #print 'pattern not in SentiWordnet'
        elif score==0:
           ncount_obj+=1 
           #print 'pattern is objective'
        else:
           ncount_sub+=1 
           polarity+=score
##    print 'Number of patterns not found:%d'%(ncount_notf)
##    print 'Number of patterns which are objective:%d'%(ncount_obj)
##    print 'Number of patterns which are subjective:%d'%(ncount_sub)
    return polarity
        
class senti_wordnet:
    def __init__(self):
        print 'Initializing SentiWordnet'
        self._temp=ddict(list)
        fswn=open('E:\\Study_Material\\quarter_2\\SNLP\\project_data\\SentiWordNet_3.0.0.txt','r')
        swn_lines=[line for line in fswn.readlines()]
        fswn.close()
        for i,line in enumerate(swn_lines):
            #print i
            data=line.strip().split('\t')
            score=(float(data[2]),float(data[3]))
            line_words=data[4].strip().split(' ')
            for wrd in line_words:
                w_n=wrd.split('#')
                w_n[0] += '#'+data[0]
                #print w_n[0]
                index=int(w_n[1])-1
                if len(self._temp[w_n[0]])>0:
                     #some lines of code
                     if (len(self._temp[w_n[0]])<=index):
                         #print 'inside if'
                         tmp_lst1=self._temp[w_n[0]]
                         tmp_lst2=tmp_lst1+[(0.0,0.0)]*(index-len(tmp_lst1)+1)
                         self._temp[w_n[0]]=tmp_lst2
                     self._temp[w_n[0]][index]=score
                else:
                     #print 'inside else'
                     #some lines of code
                     tmp_lst3=[(0.0,0.0)]*index
                     tmp_lst3.append(score)
                     self._temp[w_n[0]]=tmp_lst3
                     
    def get_score1(self,p):
        tmp_lst1=self._temp[p]
        #print tmp_lst1
        if len(tmp_lst1)==0:
            score=-1
        else:
            tmp_lst2=[]
            for tp in tmp_lst1:
                if ((1-tp[0]-tp[1])<0.75):
                    #tmp_lst2.append(tp[0]-tp[1])
                    if tp[0]>tp[1]:
                        tmp_lst2.append(tp[0])
                    else:
                        tmp_lst2.append(-tp[1])
                else:
                    tmp_lst2.append(0.0)
            score=0.0
            sumi=0.0
            flag=0
            for i,val in enumerate(tmp_lst2):
                if val!=0.0:
                    flag=1
                    score+=(1.0/(i+1))*val
                    sumi+=(1.0/(i+1))
            if flag==1:        
                score=score/sumi
##        if score!=-1 and score!=0.0:
##            print '------------------'
##            print 'score:%f'%(score)
##            print '------------------'
        return score

    def get_score2(self,p):
        tmp_lst1=self._temp[p]
        #print tmp_lst1
        if len(tmp_lst1)==0:
           p=p.split('#')[0]+'#'+'v'
           tmp_lst1=self._temp[p]
           if len(tmp_lst1)==0:
              p=p.split('#')[0]+'#'+'r'
              tmp_lst1=self._temp[p]
        if len(tmp_lst1)==0:
            score=-1
        else:
            tmp_lst2=[]
            for tp in tmp_lst1:
                if ((1-tp[0]-tp[1])<0.75):
                    #tmp_lst2.append(tp[0]-tp[1])
                    if tp[0]>tp[1]:
                        tmp_lst2.append(tp[0])
                    else:
                        tmp_lst2.append(-tp[1])
                else:
                    tmp_lst2.append(0.0)
            score=0.0
            sumi=0.0
            flag=0
            for i,val in enumerate(tmp_lst2):
                if val!=0.0:
                    flag=1
                    score+=(1.0/(i+1))*val
                    sumi+=(1.0/(i+1))
            if flag==1:        
                score=score/sumi
##        if score!=-1 and score!=0.0:
##            print '------------------'
##            print 'score:%f'%(score)
##            print '------------------'
        return score    

            
main()
fpatterns_pos.close()
fpatterns_neg.close()


        
            
