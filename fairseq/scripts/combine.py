import sys
def combine(inputFile1, inputFile2, output):

    with open(input1) as xh:
        with open(input2) as yh:
            with open(output,"w") as zh:
            #Read first file
                xlines = xh.readlines()
            #Read second file
                ylines = yh.readlines()
                count = 0
            #Combine content of both lists  and Write to third file
                for line1, line2 in zip(ylines, xlines):
                    if(line2=='\n' and line1=='\n'):
                        zh.write("\n")
                        continue
                    zh.write("{} {}\n".format(line2.rstrip(), line1.rstrip()))
                    count += 1

if __name__ == "__main__":

    input1 = sys.argv[1]
    input2 = sys.argv[2]
    output = sys.argv[3]
    combine(input1,input2,output)
