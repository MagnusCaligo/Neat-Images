import numpy as np
import cv2
import neat

srcImg = cv2.imread("lena.png")
groundTruth = cv2.Canny(srcImg, 100, 200)
#groundTruth = srcImg

w, h, c = srcImg.shape

def getRegionAroundPoint(img, x,y):
    kernelSize = 3
    pad = (kernelSize - 1)/2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((kernelSize, kernelSize, 3), np.uint8)

    for yI in range(y - pad+ 1, y + pad + 2):
        for xI in range(x - pad + 1, x + pad + 2):
            output[yI - y - pad, xI - x- pad] = img[yI, xI]
            #print img[yI, xI]

    return output

    
    

def generateCanny(geneomes, config):
    for geneome_id, geneome in geneomes:
        print "Running organism..."
        timeAlive = w * h * 2
        print "Total time", timeAlive
        width, height = groundTruth.shape

        outputImg = np.zeros((width, height, 1), np.uint8)
        net = neat.nn.FeedForwardNetwork.create(geneome, config)

        xLoc = 0
        yLoc = 0
        while timeAlive >= 0:
            cv2.imshow("Test", outputImg)
            cv2.imshow("canny", groundTruth)
            cv2.waitKey(1)
            print timeAlive
            timeAlive -= 1
            inputs = [xLoc, yLoc]
            pixels = getRegionAroundPoint(srcImg, xLoc, yLoc)
            for v in pixels:
                for m in v:
                    for t in m:
                        inputs.append(t)
            output = net.activate(inputs)
            if output[0] > .5:
                outputImg[yLoc, xLoc] = 1

            index= output.index(max(output[1:]))
            if index == 1:
                yLoc -= 1
            elif index == 2:
                xLoc += 1
            elif index == 3:
                yLoc += 1
            elif index == 4:
                xLoc -= 1

            if xLoc < 0:
               xLoc = 0
            elif xLoc >= w:
               xLoc = w -1

            if yLoc < 0:
               yLoc = 0
            elif yLoc >= h:
               yLoc = h -1

        differenceArray = np.subtract(groundTruth, outputImg)
        differenceArray = np.square(differenceArray)
        differenceValue = np.sum(differenceArray)
        fitness = 1 - differenceValue
        geneome.fitness = fitness





config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 "config-feedforward")

population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

winner = population.run(generateCanny, 300)
generateCanny([[1, 2]], None)

while True:
    cv2.imshow("Test", groundTruth)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
