# ---------------------------------------------------------------- #
# from mlxtend.plotting import plot_confusion_matrix
# ---------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np
import Config
class plot_trainning_history:
    def plot(H):
        import time
        time_now = time.localtime(time.time())
        time_save = str(time_now[0])+'_'+str(time_now[1])+str(time_now[2])+'_'+str(time_now[3])+str(time_now[4])

        print('#--------Plot-------#')
        '''plot'''
        N = Config.EPOCHS
        metrics =['loss','accuracy','precision','recall']
        for n, metric in enumerate(metrics):
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, N), H.history[metric],label='train_'+metric)
            plt.plot(np.arange(0, N), H.history['val_'+metric],label='val_'+metric)
            plt.title('Train/Validation '+metric)
            plt.xlabel('Epoch #')
            plt.ylabel(metric)
            plt.legend()
            print('[INFO]::saving plot..'+metric)
            plt.savefig(Config.PATH_PLOT+str(time_save)+'_'+metric+'.jpg')

class plot_predictions:
    def plot_image(i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(Config.CLASS_NAME[predicted_label],
                                        100*np.max(predictions_array),
                                        Config.CLASS_NAME[true_label]),
                                        color=color)

    def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(Config.CLASSES))
        plt.yticks([])
        thisplot = plt.bar(range(Config.CLASSES), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')