import numpy as np
import matplotlib as plt

# Shows the images of character at particular index of data
def show_char(data, index):
    plt.imshow(data[index, 9:439].reshape([33,13]), cmap='Greys_r')
    
# Shows the images of char from data, up to number of images shown
def show_chars_like(data, char, number):
    selected_data = np.array([x for x in data if x[1] == char])
    entries = min(number, selected_data.shape[0])
    for i in range(entries):
        show_char(selected_data, i)
        plt.show()
