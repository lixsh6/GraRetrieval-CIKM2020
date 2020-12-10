import os

train_addr = '../../../ad-hoc-udoc/train/'
test_addr = '../../../ad-hoc-udoc/test/'

def load_testQueries():
    testFiles = os.listdir(test_addr)
    testQ = []
    for testFile in testFiles:
        for i,line in enumerate(open(os.path.join(test_addr,testFile))):
            if i > 0:
                elements = line.strip().split('\t')
                qid,query = elements[0][1:],elements[2]
                testQ.append((qid,query))
                break
    return testQ

def searchCover(testQ):
    trainQids = set(map(lambda t:t.split('.')[0],os.listdir(train_addr)))
    testQids = map(lambda t:str(t[0]),testQ)
    count = 0
    for testqid in testQids:
        if testqid in trainQids:
            count += 1

    print 'count: ',count
    print 'total: ',len(testQids)
    print 'Rate: ', (float(count)/len(testQids))

def main():
    testQ = load_testQueries()
    searchCover(testQ)


if __name__ == '__main__':
    main()





