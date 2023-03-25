# GPT4-CNN-Starter

## Prerequisites
To use the code in this repository, you need to have the following installed:

- Python 3.7 or later
- PyTorch 1.8.0 or later
- torchvision 0.9.0 or later
- Pillow

## Dataset
The code assumes you have a dataset of images organized in folders, with one folder per class. For example:

```
datasets/
    train/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    test/
        class1/
            img5.jpg
            img6.jpg
        class2/
            img7.jpg
            img8.jpg
```

Replace the dataset paths in the train_data and test_data variables in the code with the paths to your own dataset folders:

```
train_data = CustomImageFolder(root='datasets/train', transform=transform)
test_data = CustomImageFolder(root='datasets/test', transform=transform)
```

## Training the Model
To train the SimpleCNN model on your dataset, execute the cells in the Jupyter Notebook. The notebook will load and preprocess the dataset, create the model architecture, and train the model for a specified number of epochs.

## Evaluating the Model
After training, the notebook will evaluate the model's performance on the test dataset and print the test accuracy.

## Using the Model
To use the trained model for image classification, load the saved model and run it on a preprocessed image:

```
model = torch.load("model.pth")

## Preprocess your image using the same transforms as the training dataset
input_image = preprocess_image("path/to/your/image.jpg", transform)
```

## Run the model on the preprocessed image

output = model(input_image.unsqueeze(0))


## Get the predicted class index

_, predicted_class_index = torch.max(output, 1)


## Map the predicted class index to the corresponding class name

```
predicted_class_name = train_data.classes[predicted_class_index.item()]
```

Replace `"path/to/your/image.jpg"` with the path to the image you want to classify. The `preprocess_image` function should apply the same transforms used during training. The output will be the predicted class name for the given input image.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This SimpleCNN model was generated with the help of OpenAI's GPT-4.
- The model serves as a starting point for more advanced image classification models and demonstrates the potential of AI-generated code in real-world applications.

