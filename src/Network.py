import numpy as np

from Func import Func

class Network:
    def __init__(self):
        
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = Func().sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = Func().sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = Func().sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000 # сколько раз пройти по всему набору данных 

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Прямой проход (эти значения нам понадобятся позже)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = Func().sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = Func().sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = Func().sigmoid(sum_o1)
                y_pred = o1

               
                d_L_d_ypred = -2 * (y_true - y_pred)

                
                d_ypred_d_w5 = h1 * Func().deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * Func().deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = Func().deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * Func().deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * Func().deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * Func().deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * Func().deriv_sigmoid(sum_h1)
                d_h1_d_b1 = Func().deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * Func().deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * Func().deriv_sigmoid(sum_h2)
                d_h2_d_b2 = Func().deriv_sigmoid(sum_h2)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем полные потери в конце каждой эпохи
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = Func().mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
