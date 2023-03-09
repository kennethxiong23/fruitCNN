import subprocess
fruitsList = subprocess.run(["ls", 'images'], capture_output=True, text=True).stdout.strip().split("\n")
for fruit in fruitsList:
    fruitpath = 'images/%s' %fruit
    fruits = subprocess.run(["ls", fruitpath], capture_output=True, text=True).stdout.strip().split("\n")
    i = 0
    for fru in fruits:
        if i <= (len(fruits) - 1500):
            remove = fruitpath + "/%s" %fru
            print(fru)
            subprocess.run(["rm", remove])
            print(i)
        i += 1 
            

