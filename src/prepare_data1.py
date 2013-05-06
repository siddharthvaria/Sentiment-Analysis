#! usr/bin/env python
#process product data
dirr='E:\\Study_Material\\quarter_2\\SNLP\project_data\\customer review data\\'
dirw=['E:\\Study_Material\\quarter_2\\SNLP\\project_data\\pos\\','E:\\Study_Material\\quarter_2\\SNLP\\project_data\\neg\\']
def main():
    num_products=5
    no_ppr=0
    no_npr=0
    for i in xrange(num_products):
        fpd=open(dirr+'product'+str(i+1)+'.txt','r')
        data=fpd.read()
        fpd.close()
        reviews=data.strip().split('[t]')
        del reviews[0]
        for review in reviews:
            score=0
            review_content=''
            review_sents=review.split('\n')
            del review_sents[0] #title of the review
            for sent in review_sents:
                if len(sent) > 0:
                    sent=sent.strip()
                    temp=sent.split('[+')
                    del temp[0]
                    #print temp
                    score1=0
                    if len(temp)>0:
                        score1=sum([int(t[0]) for t in temp if len(t) > 0])
                    temp=sent.split('[-')
                    del temp[0]
                    #print temp
                    score2=0
                    if len(temp)>0:
                        score2=sum([int(t[0]) for t in temp if len(t) > 0])
                    score+=score1-score2
                    sent=sent.split('##')
                    review_content+=sent[len(sent)-1].strip()+'\n'
            if score>0:
               fr=open(dirw[0]+'pr'+str(no_ppr)+'.txt','w')
               fr.write(review_content)
               fr.close()
               no_ppr+=1
            elif score<0:
               fr=open(dirw[1]+'pr'+str(no_npr)+'.txt','w')
               fr.write(review_content)
               fr.close()
               no_npr+=1
        print 'Product %d processed'%(i+1)

main()            
print 'Product review files created!'        
            
        
