import subprocess
fruitsList = subprocess.run(["ls", 'images'], capture_output=True, text=True).stdout.strip().split("\n")
for fruit in fruitsList:
    fruitpath = 'images/%s' %fruit
    fruits = subprocess.run(["ls", fruitpath], capture_output=True, text=True).stdout.strip().split("\n")
    for i in range(len(fruits)):
        fruit = fruit.replace(" ", "_")
        oldName = fruitpath + "/%s" %fruits[i]
        newName = fruitpath + "/%s" %(str(i)+fruit + ".png")
        subprocess.run(["mv", oldName, newName])