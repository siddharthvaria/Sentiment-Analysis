#! /usr/bin/env python
import nltk
from collections import defaultdict as ddict
common_dir1='E:\\Study_Material\\quarter_2\\SNLP\\project_data\\scale_data\\scaledata\\'
author_dir='author1\\'
common_dir2='E:\\Study_Material\\quarter_2\\SNLP\project_data\\scale_whole_review\\scale_whole_review\\'+author_dir+'txt.parag\\'

def main():
    swn=senti_wordnet()
    print 'Reading Necessary Files...'
    with open(common_dir1+author_dir+'id.txt','r') as f:
            ids=[i.strip() for i in f.readlines() if len(i.strip())>0]
    with open(common_dir1+author_dir+'label4.txt','r') as f:
            labels=[int(label.strip()) for label in f.readlines() if len(label.strip())>0]
    with open(common_dir1+author_dir+'subj.txt','r') as f:
            review_summaries=[review.strip() for review in f.readlines() if len(review.strip())>0]
    polarities=[]
    #label assumes one of the four values 0,1,2,3
    for label in labels:
        if label==0 or label==1:
            polarities.append('neg')
        else:
            polarities.append('pos')
    wraccuracy_neg=0
    rsaccuracy_neg=0
    wraccuracy_pos=0
    rsaccuracy_pos=0
    fwr=open('wrresults.txt','w')
    frs=open('rsresults.txt','w')
    num_reviews=200

    # for negative reviews
    '''
    ids_partial=ids[:num_reviews]
    polarities=polarities[:num_reviews]
    review_summaries=review_summaries[:num_reviews]
    '''
    # for positive reviews
    ids_partial=ids[-num_reviews:]
    polarities=polarities[-num_reviews:]
    review_summaries=review_summaries[-num_reviews:]
    
    total_neg_reviews=polarities.count('neg')
    total_pos_reviews=polarities.count('pos')
    print 'Total negative reviews:%d'%(total_neg_reviews)
    print 'Total positive reviews:%d'%(total_pos_reviews)
    print 'Starting to process approximately %d reviews.It may take few minutes...'%(len(ids_partial))
    for i,fileid in enumerate(ids_partial):
        with open(common_dir2+fileid+'.txt','r') as f:
                whole_review_parags=[parag.strip() for parag in f.readlines() if len(parag.strip())>0]
        review_summary=review_summaries[i]
        review_summary_lines=[line.strip()+'.' for line in review_summary.split('.')]
        whole_review_lines=[]      
        for parag in whole_review_parags:
            if ('director:' in parag or 'Director:' in parag or 'director/writer:' in parag or 'Director/Writer:' in parag) and ('cast:' in parag or 'Cast:' in parag):
                continue;
            elif 'Reviewed by ' in parag:
                continue;
            elif 'REVIEWED ON ' in parag:
                break;
            else:
                whole_review_lines+=[line.strip()+'.' for line in parag.split('.')]
                         
        rspol=extract_features_n_get_polarity(review_summary_lines,swn)
        wrpol=extract_features_n_get_polarity(whole_review_lines,swn)
        if polarities[i]=='pos':
                if rspol > 0:
                    rsaccuracy_pos+=1
                else:
                    frs.write('Review:%d\tID:%d\tActual Polarity:%s\tCalculated Polarity:%f\n'%(i,int(fileid),polarities[i],rspol))
                if wrpol > 0:
                    wraccuracy_pos+=1
                else:
                    fwr.write('Review:%d\tID:%d\tActual Polarity:%s\tCalculated Polarity:%f\n'%(i,int(fileid),polarities[i],wrpol))
        elif polarities[i]=='neg':
                if rspol < 0:
                    rsaccuracy_neg+=1
                else:
                    frs.write('Review:%d\tID:%d\tActual Polarity:%s\tCalculated Polarity:%f\n'%(i,int(fileid),polarities[i],rspol))
                if wrpol < 0:
                    wraccuracy_neg+=1
                else:
                    fwr.write('Review:%d\tID:%d\tActual Polarity:%s\tCalculated Polarity:%f\n'%(i,int(fileid),polarities[i],wrpol))
        if i%25==0:
           print '. ',
    #fwr.write('Accuracy of negative reviews:%f\n'%((float(wraccuracy_neg)*100)/total_neg_reviews))
    fwr.write('Accuracy of positive reviews:%f\n'%((float(wraccuracy_pos)*100)/total_pos_reviews))
    #frs.write('Accuracy of negative reviews:%f\n'%((float(rsaccuracy_neg)*100)/total_neg_reviews))
    frs.write('Accuracy of positive reviews:%f\n'%((float(rsaccuracy_pos)*100)/total_pos_reviews))
    fwr.close()
    frs.close()
    
def extract_features_n_get_polarity(ls,swn):
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
            if ((tg[0][1]=='RB' or tg[0][1]=='RBR' or tg[0][1]=='RBS') and (tg[1][1]=='VB' or tg[1][1]=='VBG' or tg[1][1]=='VBN' or tg[1][1]=='VBD')):
                patterns.append(tg[0][0].lower()+'#'+'r')
    polarity=0
    ncount_obj=0
    ncount_sub=0
    ncount_notf=0
    for p in patterns:
        score=swn.get_score2(p)
        if score==-1:
           ncount_notf+=1
##           if directory=='pos':
##               fpatterns_pos.write(p+'\n')
##           else:
##               fpatterns_neg.write(p+'\n')
##           print 'pattern not in SentiWordnet'
        elif score==0:
           ncount_obj+=1 
           #print 'pattern is objective'
        else:
           #ncount_sub+=1 
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
        swn_lines=[line.strip() for line in fswn.readlines()]
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
                 p=p.split('#')[0]+'#'+'n'
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
