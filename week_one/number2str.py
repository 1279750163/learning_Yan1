if __name__ == '__main__':

    with open('lenses.txt') as f:
        dataset = []
        for line in f.readlines():
            data = []
            line = [a for a in line.split()]
            print(line)
            print(line[1])
            if line[1] == '1':
                data.append('young')
            if line[1] == '2':
                data.append('pre-presbyopic')
            if line[1] == '3':
                data.append('presbyopic')
            if line[2] == '1':
                data.append('myope')
            if line[2] == '2':
                data.append('hypermetrope')
            if line[3] == '1':
                data.append('no')
            if line[3] == '2':
                data.append('yes')
            if line[4] == '1':
                data.append('reduced')
            if line[4] == '2':
                data.append('normal')
            if line[5] == '1':
                data.append('hard')
            if line[5] == '2':
                data.append('soft')
            if line[5] == '3':
                data.append('no_lenses')
            dataset.append(data)
    print(dataset)
    with open('ID3.txt', 'w') as f:
        for i in dataset:
            i = ' '.join(i)
            f.write(i + '\n')
