#! /usr/bin/env python
import nltk
from nltk.probability import MLEProbDist, WittenBellProbDist 
from collections import defaultdict as ddict

class MostLikelyTagModel(object):
    def __init__(self):
        self._word_tags = {}

    def train(self, tagged_sentences):
        # count number of times a word is given each tag
        word_tag_counts = ddict(lambda: ddict(lambda: 0))
        for sent in tagged_sentences:
            for (word, tag) in sent:
                word_tag_counts[word][tag] += 1
        # select the tag used most often for the word
        for word in word_tag_counts:
            tag_counts = word_tag_counts[word]
            tag = max(tag_counts, key=tag_counts.get)
            self._word_tags[word] = tag

    def predict(self, sentence):
        # predict the tags for each word in the sentence,
        # using the most common tag, or NN if never seen
        return [self._word_tags.get(word, 'NN') for word in sentence]

    def prediction(self, sentence):
        # string showing prediction
        tags = self.predict(sentence)
        tagged = [word + '/' + tag for (word, tag) in zip(sentence,tags)]
        return ' '.join(tagged)
    
    def get_error(self, tagged_sentences):
        # get word error rate
        word_tuples = self._get_word_tuples(tagged_sentences)
        return self._get_error(word_tuples)

    def get_known_unknown_error(self, tagged_sentences):
        # split predictions into known and unknown words
        known = []
        unknown = []
        for tup in self._get_word_tuples(tagged_sentences):
            word, _, _ = tup
            dest = known if word in self._word_tags else unknown
            dest.append(tup)
        # calculate and return known and unknown error rates
        return self._get_error(known), self._get_error(unknown)

    def get_confusion_matrix(self, tagged_sentences):
        # get confusion matrix for gold, predicted tags
        word_tuples = self._get_word_tuples(tagged_sentences)
        gold = [tag for (_, tag, _) in word_tuples]
        test = [tag for (_, _, tag) in word_tuples]
        return nltk.ConfusionMatrix(gold, test)               
        
    def _get_word_tuples(self, tagged_sentences):
        # convert a list of sentences into word-tag tuples
        word_tuples = []
        for sent in tagged_sentences:
            words = [word for (word, _tag) in sent]
            predicted = self.predict(words)
            for i, (word, tag) in enumerate(sent):
                word_tuples.append((word, tag, predicted[i]))
        return word_tuples

    def _get_error(self, word_tuples):
        # calculate total and incorrect labels
        incorrect = 0
        for word, expected_tag, actual_tag in word_tuples:
            if expected_tag != actual_tag:
                incorrect += 1
        return 1.0 * incorrect / len(word_tuples)


class HMM(MostLikelyTagModel):
    INITIAL = '<s>'
    FINAL = '</s>'
    UNK = '<unk>'

    def __init__(self, use_final_transition=True):
        super(HMM, self).__init__()
        self.use_final_transition = use_final_transition
        self._states = []
        self._transitions = ddict(lambda: ddict(lambda: 0.0))
        self._emissions = ddict(lambda: ddict(lambda: 0.0))

    def add(self, state, transition_dict={}, emission_dict={}):
        self._states.append(state)
        # build state transition matrix
        for target_state, prob in transition_dict.items():
            self._transitions[state][target_state] = prob
        # build observation emission matrix
        for observation, prob in emission_dict.items():
            self._emissions[state][observation] = prob
        # if no final state transition, just add with probability 1
        if not self.use_final_transition:
            self._transitions[state][self.FINAL] = 1.0

    def probability(self, observations):
        # get probability from the last entry in the trellis
        probs = self._forward(observations)
        return probs[len(observations)][self.FINAL]

    def _forward(self, observations):
        # initialize the trellis
        probs = ddict(lambda: ddict(lambda: 0.0))
        probs[-1][self.INITIAL] = 1.0
        # update the trellis for each observation
        for i, observation in enumerate(observations):
            for state in self._states:
                # sum the probabilities of transitioning to
                # the current state and emitting the current
                # observation from any of the previous states
                probs[i][state] = sum(
                    probs[i - 1][prev_state] *
                    self._transition(prev_state, state) *
                    self._emission(state, observation)
                    for prev_state in self._states)
        # sum the probabilities for all states in the last
        # column (the last observation) of the trellis,
        # transitioning to the final state
        probs[len(observations)][self.FINAL] = sum(
            probs[i][state] *
            self._transition(state, self.FINAL)
            for state in self._states)
        return probs

    def _transition(self, prev_state, state):
        return self._transitions[prev_state][state]
        
    def _emission(self, state, observation):
        # use '<unk>' if observation not seen
        if observation not in self._emissions[state]:
            return self._emissions[state][self.UNK]
        else: return self._emissions[state][observation]
        
    def predict(self, observations):
        # initialize the probabilities and backpointers for viterbi
        probs = ddict(lambda: ddict(lambda: 0.0))
        probs[-1][self.INITIAL] = 1.0
        pointers = ddict(lambda: {})
        # update the probabilities for each observation
        for i, observation in enumerate(observations):
            for state in self._states:
                # calculate probabilities of taking a transition
                # from a previous state to this one and emitting
                # the current observation
                path_probs = {}
                for prev_state in self._states:
                    path_probs[prev_state] = (
                        probs[i - 1][prev_state] *
                        self._transition(prev_state, state) *
                        self._emission(state, observation))
                # select previous state with the highest probability
                best_state = max(path_probs, key=path_probs.get)
                probs[i][state] = path_probs[best_state]
                pointers[i][state] = best_state
        # get the best final state
        path_probs = {}
        for state in self._states:
            path_probs[state] = (
                probs[i][state] *
                self._transition(state, self.FINAL))
        best_final_state = max(path_probs, key=path_probs.get)
        # follow the pointers to get the best state sequence
        states = []
        curr_state = best_final_state
        for i in xrange(i, -1, -1):
            states.append(curr_state)
            curr_state = pointers[i][curr_state]
        states.reverse()
        return states

    def train(self, tagged_sentences):
        # call super so we'll know which words are seen in training
        super(HMM, self).train(tagged_sentences)
        # get (tag, word) and (tag, tag) pairs for each sentence
        tag_word = []
        tag_tag = []
        for sent in tagged_sentences:
            tag_word.extend([(tag,word) for (word,tag) in sent])
            tags = [tag for (_word,tag) in sent]
            # add <s> and </s>
            tags.insert(0, self.INITIAL)
            tags.append(self.FINAL)
            tag_tag.extend(nltk.bigrams(tags))
        # get counts as conditional frequency distributions
        tag_word_cfd = nltk.ConditionalFreqDist(tag_word)
        tag_tag_cfd = nltk.ConditionalFreqDist(tag_tag)
        # get probabilities as conditional probability distributions
        pd_factory = lambda fd: WittenBellProbDist(fd, 50000)
        tag_word_cpd = nltk.ConditionalProbDist(tag_word_cfd, pd_factory) 
        tag_tag_cpd = nltk.ConditionalProbDist(tag_tag_cfd, MLEProbDist)
        # add HMM states using these distributions
        make_dict = lambda pd: dict([(sample, pd.prob(sample))
                                     for sample in pd.samples()])
        self.add(self.INITIAL, make_dict(tag_tag_cpd[self.INITIAL]))
        for tag in tag_word_cpd.conditions():
            tag_tag_dict = make_dict(tag_tag_cpd[tag])
            tag_word_dict = make_dict(tag_word_cpd[tag])
            tag_word_dict[self.UNK] = tag_word_cpd[tag].prob(self.UNK)
            self.add(tag, tag_tag_dict, tag_word_dict)

def get_trained_hmm():
    treebank_sents = nltk.corpus.treebank.tagged_sents()
    model = HMM()
    model.train(treebank_sents)
    sent='a mind-fuck movie for the teen generation that touches on a very cool idea , but presents it in a very bad package.'
    sent_wrds=nltk.word_tokenize(sent)
    print model.prediction(sent_wrds)
    #return model


get_trained_hmm()
