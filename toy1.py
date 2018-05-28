from seq2seq import Seq2seq

def main():
    seq2seq = Seq2seq(lr=0.02, init_range=0.3)
    data = 0
    data_test = 0
    training_accuracy = 0
    testing_accuracy = 0
    input_train = []
    output_train = []
    input_test = []
    output_test = []

    with open("train_data.txt", 'r') as td:
        while True:
            out_train = []
            lines = td.readline()
            if not lines:
                break
            data += 1
            ipd, gt = lines.split("->")
            in_train = list(map(int, ipd.split(" ")))
            out_train.append(gt)
            out_train = list(map(int, out_train))
            input_train.append(in_train)
            output_train.append(out_train)
    td.close()

    with open("test_data.txt", 'r') as ted:
        while True:
            out_test = []
            lines = ted.readline()
            if not lines:
                break
            data_test += 1
            ipd, gt = lines.split("->")
            in_test = list(map(int, ipd.split(" ")))
            out_test.append(gt)
            out_test = list(map(int, out_test))
            input_test.append(in_test)
            output_test.append(out_test)
    td.close()

    for i in range(3000):
        cost = 0
        for j in range(len(input_train)):
            cost += seq2seq.train(input_train[j], output_train[j])
            
        if i % 100 == 0:
            correct = 0
            print('Epoch:', i)
            # print('train result:')
            for k in range(len(input_train)):
               # print(input_train[k], '->', seq2seq.predict(input_train[k]))
                if seq2seq.predict(input_train[k]) == output_train[k]:
                    correct += 1
                training_accuracy = correct / data
            print('loss:', cost / data)
            print('train accuracy:', training_accuracy)

    correct_test = 0
    print('test result:')
    for l in range(len(input_test)):
        #print(input_test[l], '->', seq2seq.predict(input_test[l]))
        if seq2seq.predict(input_test[l]) == output_test[l]:
            correct_test +=1
        testing_accuracy = correct_test / data_test
    print('test accuracy', testing_accuracy)


    # for i in range(5000):
    #     cost = seq2seq.train([2], [2])
    #     cost += seq2seq.train([1], [1])
    #     cost += seq2seq.train([3], [3])
    #     cost += seq2seq.train([1, 3], [3])
    #     cost += seq2seq.train([1, 2], [2])
    #     cost += seq2seq.train([3, 2], [3])
    #     cost += seq2seq.train([1, 2, 3], [3])

    #     if i % 100 == 0:
    #         print('Epoch:', i)
    #         print('training cost:', cost / 7)

    #         print('train:')
    #         print([1], '->', seq2seq.predict([1]))
    #         print([2], '->', seq2seq.predict([2]))
    #         print([3], '->', seq2seq.predict([3]))
    #         print([1, 3], '->', seq2seq.predict([1, 3]))
    #         print([1, 2], '->', seq2seq.predict([1, 2]))
    #         print([3, 2], '->', seq2seq.predict([3, 2]))
    #         print([1, 2, 3], '->', seq2seq.predict([1, 2, 3]))
    #         print('test:')
    #         print([3, 1], '->', seq2seq.predict([3, 1]))
    #         print([2, 1], '->', seq2seq.predict([2, 1]))
    #         print([2, 3], '->', seq2seq.predict([2, 3]))
    #         print([3, 2, 1], '->', seq2seq.predict([3, 2, 1]))

if __name__ == "__main__":
    main()
