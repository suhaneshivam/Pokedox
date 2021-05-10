![Pokemon](https://www.pyimagesearch.com/wp-content/uploads/2018/04/cnn_keras_squirtle_result_01.png)


># Pokedox
***
This deep nueral net can identify the 4 different Pokemons after training for 100 epochs with an accuracy of 82%.

## Model 
SmallVggNet was used to train the network so that anyone having less sophisticated machine train model in less time.

## Dateset
To download the dataset,user have to run _download_images.py_ script from cms which is included in helper folder.You also have to pass two additional 
1) --output - indicating the directory where you want to download the dataset.
2) --urls - where urls.txt files are located.
`c:\location\of\the\python\file>python download_images.py --output output/location --urls location/of/urls.txt`

urls.txt files are included in urls folder.Additionally if you want to create own urls.txt file than:
On Chrome: Settings>More Tools>Developer Tools>Console than paste the code included in url.js file present helper directory.Then search for the Pokenmon character you want to downlaod and scroll the images according to the number of images you want and then execute the code.This is automatically download the urls.txt file for you.You can repeat the steps for every character.

## Output
![Plot](https://github.com/suhaneshivam/Pokedox/blob/main/output/plot.png?raw=true)






