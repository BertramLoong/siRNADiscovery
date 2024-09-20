import os

if __name__ == '__main__':
    path = "./"

    files = os.listdir(path)

    for file in files:

        if "_0001_dp.ps" not in file or '.bpp' in file:
            continue


        name = file.replace("_0001_dp.ps", "")


        temp = open(path + file).readlines()


        start_flag = False


        os.makedirs("RNAfold_bp_file", exist_ok=True)
        f = open("RNAfold_bp_file/" + file + ".bpp", "w")


        for line in temp:

            line = line.strip()


            if "start of base pair probability data" in line:
                start_flag = True


            if start_flag == True and "ubox" in line:

                line = line.strip().split()


                assert(len(line) == 4)


                i, j, prob, _ = line


                prob = float(prob)


                f.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")


        f.close()