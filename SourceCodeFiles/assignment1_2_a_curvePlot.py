import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xlrd

def compute_SquareError_polynomial_1(b_current, m_current, x_taken,y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += (y - (m_current * x + b_current)) ** 2
    return totalSquareError / (2* float(len(x_taken)))

def compute_SquareError_polynomial_2(b_current, m1_current, m2_current, x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += (y - (m1_current * x + m2_current*x*x + b_current)) ** 2
    return totalSquareError / (2 * float(len(x_taken)))
def compute_SquareError_polynomial_3(b_current, m1_current, m2_current, m3_current, x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+ b_current)-y  ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))
def compute_SquareError_polynomial_4(b_current, m1_current, m2_current, m3_current,m4_current, x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+ b_current)-y  ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))

def compute_SquareError_polynomial_5(b_current, m1_current, m2_current, m3_current,m4_current,m5_current, x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+(m5_current*x*x*x*x*x)+ b_current)-y  ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))
def compute_SquareError_polynomial_6(b_current, m1_current, m2_current, m3_current,m4_current,m5_current,m6_current, x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+(m5_current*x*x*x*x*x)+(m6_current*x*x*x*x*x*x)+ b_current)-y  ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))
def compute_SquareError_polynomial_7(b_current, m1_current, m2_current, m3_current,m4_current,m5_current,m6_current, m7_current,x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+(m5_current*x*x*x*x*x)+(m6_current*x*x*x*x*x*x)+(m7_current*x*x*x*x*x*x*x)+ b_current)-y ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))
def compute_SquareError_polynomial_8(b_current, m1_current, m2_current, m3_current,m4_current,m5_current,m6_current, m7_current,m8_current,x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+(m5_current*x*x*x*x*x)+(m6_current*x*x*x*x*x*x)+(m7_current*x*x*x*x*x*x*x)+(m8_current*x*x*x*x*x*x*x*x)+ b_current)-y ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))

def compute_SquareError_polynomial_9(b_current, m1_current, m2_current, m3_current,m4_current,m5_current,m6_current, m7_current,m8_current,m9_current,x_taken, y_taken):
    totalSquareError = 0
    for i in range(0, len(x_taken)):
        x = x_taken[i]
        y = y_taken[i]
        totalSquareError += ((m1_current * x + (m2_current * x * x) +(m3_current*x*x*x)+(m4_current*x*x*x*x)+(m5_current*x*x*x*x*x)+(m6_current*x*x*x*x*x*x)+(m7_current*x*x*x*x*x*x*x)+(m8_current*x*x*x*x*x*x*x*x)+(m9_current*x*x*x*x*x*x*x*x*x)+ b_current)-y ) ** 2
    return totalSquareError / (2 * float(len(x_taken)))

def gradient_descent_polynomial_x_1(x_taken,y_taken,iteration,learning_rate):


        b_current=0
        m_current=0
        er=0
        for i in range(iteration):
            b_gradient = 0
            m_gradient = 0
            N = float(len(x_taken))
            for j in range(0, len(x_taken)):
                x = x_taken[j]
                y = y_taken[j]
                #print("x and y ==",x,y)
                y_p=((m_current * x) + b_current)
                b_gradient += (y_p-y)
                m_gradient +=  x *(y_p-y)
            b_gradient = (1 / N) * b_gradient
            m_gradient = (1 / N) * m_gradient
            b_current = b_current - (learning_rate * b_gradient)
            m_current = m_current - (learning_rate * m_gradient)
            er = compute_SquareError_polynomial_1(b_current, m_current, x_taken,y_taken)
            #print("every step b===", b_current)
            #print("every step m===", m_current)
            #print("every step error==", er)
        #print("final b===",b_current)
        #print("final m===",m_current)
        #print("final error==",er)
        return[b_current,m_current,er]



def gradient_descent_polynomial_x_2(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x)+(m2_current *x * x) +  b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        er = compute_SquareError_polynomial_2(b_current, m1_current, m2_current, x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current, er]

def gradient_descent_polynomial_x_3(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)  + b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        er = compute_SquareError_polynomial_3(b_current, m1_current, m2_current, m3_current, x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,er]

def gradient_descent_polynomial_x_4(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x) + b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        er = compute_SquareError_polynomial_4(b_current, m1_current, m2_current, m3_current,m4_current, x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,er]

def gradient_descent_polynomial_x_5(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    m5_current = 0

    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        m5_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x)+(m5_current *x* x *x* x * x) + b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
            m5_gradient += x * x * x * x *x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        m5_gradient = (1 / N) * m5_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        m5_current = m5_current - (learning_rate * m5_gradient)
        er = compute_SquareError_polynomial_5(b_current, m1_current, m2_current, m3_current,m4_current, m5_current,x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,m5_current,er]

def gradient_descent_polynomial_x_6(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    m5_current = 0
    m6_current = 0

    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        m5_gradient = 0
        m6_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x)+(m5_current *x* x *x* x * x) +(m6_current *x* x *x*x* x * x)+ b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
            m5_gradient += x * x * x * x *x * (y_p - y)
            m6_gradient += x *x* x * x * x * x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        m5_gradient = (1 / N) * m5_gradient
        m6_gradient = (1 / N) * m6_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        m5_current = m5_current - (learning_rate * m5_gradient)
        m6_current = m6_current - (learning_rate * m6_gradient)
        er = compute_SquareError_polynomial_6(b_current, m1_current, m2_current, m3_current,m4_current, m5_current,m6_current,x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,m5_current,m6_current,er]


def gradient_descent_polynomial_x_7(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    m5_current = 0
    m6_current = 0
    m7_current = 0

    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        m5_gradient = 0
        m6_gradient = 0
        m7_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x)+(m5_current *x* x *x* x * x) +(m6_current *x* x *x*x* x * x)+(m7_current* x *x* x *x*x* x * x)+ b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
            m5_gradient += x * x * x * x *x * (y_p - y)
            m6_gradient += x *x* x * x * x * x * (y_p - y)
            m7_gradient += x*x * x * x * x * x * x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        m5_gradient = (1 / N) * m5_gradient
        m6_gradient = (1 / N) * m6_gradient
        m7_gradient = (1 / N) * m7_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        m5_current = m5_current - (learning_rate * m5_gradient)
        m6_current = m6_current - (learning_rate * m6_gradient)
        m7_current = m7_current - (learning_rate * m7_gradient)
        er = compute_SquareError_polynomial_7(b_current, m1_current, m2_current, m3_current,m4_current, m5_current,m6_current,m7_current,x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,m5_current,m6_current,m7_current,er]


def gradient_descent_polynomial_x_8(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    m5_current = 0
    m6_current = 0
    m7_current = 0
    m8_current = 0
    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        m5_gradient = 0
        m6_gradient = 0
        m7_gradient = 0
        m8_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x)+(m5_current *x* x *x* x * x) +(m6_current *x* x *x*x* x * x)+(m7_current* x *x* x *x*x* x * x)+(m8_current* x*x *x* x *x*x* x * x)+ b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
            m5_gradient += x * x * x * x *x * (y_p - y)
            m6_gradient += x *x* x * x * x * x * (y_p - y)
            m7_gradient += x*x * x * x * x * x * x * (y_p - y)
            m8_gradient += x *x* x * x * x * x * x * x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        m5_gradient = (1 / N) * m5_gradient
        m6_gradient = (1 / N) * m6_gradient
        m7_gradient = (1 / N) * m7_gradient
        m8_gradient = (1 / N) * m8_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        m5_current = m5_current - (learning_rate * m5_gradient)
        m6_current = m6_current - (learning_rate * m6_gradient)
        m7_current = m7_current - (learning_rate * m7_gradient)
        m8_current = m8_current - (learning_rate * m8_gradient)
        er = compute_SquareError_polynomial_8(b_current, m1_current, m2_current, m3_current,m4_current, m5_current,m6_current,m7_current,m8_current,x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,m5_current,m6_current,m7_current,m8_current,er]
def gradient_descent_polynomial_x_9(x_taken,y_taken,iteration,learning_rate):
    b_current = 0
    m1_current = 0
    m2_current = 0
    m3_current = 0
    m4_current=0
    m5_current = 0
    m6_current = 0
    m7_current = 0
    m8_current = 0
    m9_current = 0
    er = 0
    for i in range(iteration):
        b_gradient = 0
        m1_gradient = 0
        m2_gradient = 0
        m3_gradient=0
        m4_gradient=0
        m5_gradient = 0
        m6_gradient = 0
        m7_gradient = 0
        m8_gradient = 0
        m9_gradient = 0
        N = float(len(x_taken))
        for j in range(0, len(x_taken)):
            x = x_taken[j]
            y = y_taken[j]
            # print("x and y ==",x,y)
            y_p = ((m1_current * x) + (m2_current * x * x)+(m3_current * x * x * x)+(m4_current *x* x * x * x)+(m5_current *x* x *x* x * x) +(m6_current *x* x *x*x* x * x)+(m7_current* x *x* x *x*x* x * x)+(m8_current* x*x *x* x *x*x* x * x)+(m9_current*x* x*x *x* x *x*x* x * x)+ b_current)
            b_gradient += (y_p - y)
            m1_gradient += x * (y_p - y)
            m2_gradient += x * x * (y_p - y)
            m3_gradient += x * x *x* (y_p - y)
            m4_gradient += x * x * x *x* (y_p - y)
            m5_gradient += x * x * x * x *x * (y_p - y)
            m6_gradient += x *x* x * x * x * x * (y_p - y)
            m7_gradient += x*x * x * x * x * x * x * (y_p - y)
            m8_gradient += x *x* x * x * x * x * x * x * (y_p - y)
            m9_gradient += x * x * x * x * x * x * x * x * x * (y_p - y)
        b_gradient = (1 / N) * b_gradient
        m1_gradient = (1 / N) * m1_gradient
        m2_gradient = (1 / N) * m2_gradient
        m3_gradient = (1 / N) * m3_gradient
        m4_gradient=(1 / N) * m4_gradient
        m5_gradient = (1 / N) * m5_gradient
        m6_gradient = (1 / N) * m6_gradient
        m7_gradient = (1 / N) * m7_gradient
        m8_gradient = (1 / N) * m8_gradient
        m9_gradient = (1 / N) * m8_gradient
        b_current = b_current - (learning_rate * b_gradient)
        m1_current = m1_current - (learning_rate * m1_gradient)
        m2_current = m2_current - (learning_rate * m2_gradient)
        m3_current = m3_current - (learning_rate * m3_gradient)
        m4_current = m4_current - (learning_rate * m4_gradient)
        m5_current = m5_current - (learning_rate * m5_gradient)
        m6_current = m6_current - (learning_rate * m6_gradient)
        m7_current = m7_current - (learning_rate * m7_gradient)
        m8_current = m8_current - (learning_rate * m8_gradient)
        m9_current = m9_current - (learning_rate * m9_gradient)
        er = compute_SquareError_polynomial_9(b_current, m1_current, m2_current, m3_current,m4_current, m5_current,m6_current,m7_current,m8_current,m9_current,x_taken, y_taken)
        # print("every step b===", b_current)
        # print("every step m===", m_current)
        # print("every step error==", er)
    # print("final b===",b_current)
    # print("final m===",m_current)
    # print("final error==",er)
    return [b_current, m1_current, m2_current,m3_current,m4_current,m5_current,m6_current,m7_current,m8_current,m9_current,er]




df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\sampleData.xls","sheet 1")

#newData=rd.shuffle(df)

trainPer=int((len(df))*(70/100))
#print("trainper==",trainPer)
train_data = df[:trainPer]
test_data =df[trainPer:]
print("train== \n",train_data)
print("test== \n",test_data)
x=[]
y=[]
total_dataX=[]
total_dataY=[]
data_ar=np.array(train_data)
var=0
for i in range(0, len(data_ar)):
    x.insert(i,data_ar[i, 0])
    y.insert(i,data_ar[i, 1])
    total_dataX.insert(var,data_ar[i,0])
    total_dataY.insert(var, data_ar[i, 1])
    var=var+1
x_test=[]
y_test=[]
data_test=np.array(test_data)
for i in range(0, len(data_test)):
    x_test.insert(i,data_test[i, 0])
    y_test.insert(i,data_test[i, 1])
    total_dataX.insert(var, data_test[i, 0])
    total_dataY.insert(var, data_test[i, 1])
    var = var + 1
print("x_test==",x_test)
print("y_test==",y_test)
print("totalDatax==",total_dataX)
print("totalDataY==",total_dataY)


plt.xlabel('X')
plt.ylabel('Y')
plt.title("Separate Data Plot")

plt.plot(total_dataX,total_dataY, color='blue',marker='X')
plt.show()

learning_rate=0.05
iteration=2000
#polynomial n=1 ------------------
[b_1,m_1,er_1]=gradient_descent_polynomial_x_1(x,y,iteration,learning_rate)
print("For polynomial n=1 estimated W : m {}, b {},squareError {}".format(m_1,b_1,er_1))
test_er1=compute_SquareError_polynomial_1(b_1, m_1, x_test,y_test)
print("test error on test data for polynomial n==1 is ==",test_er1)
y_predict_poly1=[]
for i in range(len(total_dataX)):
    y_predict_poly1.insert(i, (b_1+m_1*total_dataX[i]))
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=1")
plt.scatter(total_dataX,total_dataY, color='red', marker= '*')
plt.plot(total_dataX,y_predict_poly1, color='blue')
plt.show()
#polynomial n=2 ------------------
[b_2,m_2_1,m_2_2,er_2]=gradient_descent_polynomial_x_2(x,y,iteration,learning_rate)
print("For polynomial n=2 estimated W : m_2_1 {}, m_2_2 {} , b {},squareError {}".format(m_2_1,m_2_2,b_2,er_2))
test_er2=compute_SquareError_polynomial_2(b_2, m_2_1, m_2_2, x_test,y_test)
print("test error on test data for polynomial n==2 is ==",test_er2)


y_predict_poly2=[]
for i in range(len(total_dataX)):

    y_predict_poly2.insert(i, b_2+m_2_1*total_dataX[i]+m_2_2*total_dataX[i]*total_dataX[i])
#print(y_predict_poly1
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=2")
plt.scatter(total_dataX,total_dataY, color='red', marker= '*')
plt.plot(total_dataX,y_predict_poly2, color='blue')
plt.show()
#polynomial n=3 ------------------
[b_3,m_3_1,m_3_2,m_3_3,er_3]=gradient_descent_polynomial_x_3(x,y,iteration,learning_rate)
print("For polynomial n=3 estimated W : m_3_1 {}, m_3_2 {} ,m_3_3{}, b {},squareError {}".format(m_3_1,m_3_2,m_3_3,b_3,er_3))
test_er3=compute_SquareError_polynomial_3(b_3, m_3_1, m_3_2, m_3_3,x_test,y_test)
print("test error on test data for polynomial n==3 is ==",test_er3)
y_predict_poly3=[]
for i in range(len(total_dataX)):
    y_predict_poly3.insert(i, b_3+m_3_1*total_dataX[i]+m_3_2*total_dataX[i]*total_dataX[i]+m_3_3*total_dataX[i]*total_dataX[i]*total_dataX[i])
#print(y_predict_poly1)
plt.xlabel('X ')
plt.ylabel('Y')
plt.title("Fitted curve for polynomial n=3")
plt.scatter(total_dataX,total_dataY, color='red', marker= '*')
plt.plot(total_dataX,y_predict_poly3, color='blue')
plt.show()
#polynomial n=4 ------------------
[b_4,m_4_1,m_4_2,m_4_3,m_4_4,er_4]=gradient_descent_polynomial_x_4(x,y,iteration,learning_rate)
print("For polynomial n=4 estimated W : m_4_1 {}, m_4_2 {} ,m_4_3 {},m_4_4 {} , b_4 {},squareError {}".format(m_4_1,m_4_2,m_4_3,m_4_4,b_4,er_4))
test_er4=compute_SquareError_polynomial_4(b_4, m_4_1, m_4_2, m_4_3,m_4_4, x_test,y_test)
print("test error on test data for polynomial n==4 is ==",test_er4)
y_predict_poly4=[]
for i in range(len(df.x)):
    #print("print data ------######--{}", df.x[i])
    y_predict_poly4.insert(i, b_4+m_4_1*df.x[i]+m_4_2*df.x[i]*df.x[i]+m_4_3*df.x[i]*df.x[i]*df.x[i]+m_4_4*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=4")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly4, color='blue')
plt.show()
#polynomial n=5 ------------------
[b_5,m_5_1,m_5_2,m_5_3,m_5_4,m_5_5,er_5]=gradient_descent_polynomial_x_5(x,y,iteration,learning_rate)
print("For polynomial n=5 estimated W : m_5_1 {}, m_5_2 {} ,m_5_3 {},m_5_4 {}, m_5_5 {} , b_5 {},squareError {}".format(m_5_1,m_5_2,m_5_3,m_5_4,m_5_5,b_5,er_5))
test_er5=compute_SquareError_polynomial_5(b_5, m_5_1, m_5_2, m_5_3,m_5_4,m_5_5, x_test,y_test)
print("test error on test data for polynomial n==5 is ==",test_er5)
y_predict_poly5=[]
for i in range(len(df.x)):
    y_predict_poly5.insert(i, b_5+m_5_1*df.x[i]+m_5_2*df.x[i]*df.x[i]+m_5_3*df.x[i]*df.x[i]*df.x[i]+m_5_4*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_5_5*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=5")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly5, color='blue')
plt.show()
#polynomial n=6 ------------------
[b_6,m_6_1,m_6_2,m_6_3,m_6_4,m_6_5,m_6_6,er_6]=gradient_descent_polynomial_x_6(x,y,iteration,learning_rate)
print("For polynomial n=6 estimated W : m_6_1 {}, m_6_2 {} ,m_6_3 {},m_6_4 {}, m_6_5 {} , m_6_6 {}, b_6 {},squareError {}".format(m_6_1,m_6_2,m_6_3,m_6_4,m_6_5,m_6_6,b_6,er_6))
test_er6=compute_SquareError_polynomial_6(b_6, m_6_1, m_6_2, m_6_3,m_6_4,m_6_5,m_6_6, x_test,y_test)
print("test error on test data for polynomial n==6 is ==",test_er6)
y_predict_poly6=[]
for i in range(len(df.x)):
    y_predict_poly6.insert(i, b_6+m_6_1*df.x[i]+m_6_2*df.x[i]*df.x[i]+m_6_3*df.x[i]*df.x[i]*df.x[i]+m_6_4*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_6_5*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_6_6*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=6")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly6, color='blue')
plt.show()
#polynomial n=7 ------------------
[b_7,m_7_1,m_7_2,m_7_3,m_7_4,m_7_5,m_7_6,m_7_7,er_7]=gradient_descent_polynomial_x_7(x,y,iteration,learning_rate)
print("For polynomial n=7 estimated W : m_7_1 {}, m_7_2 {} ,m_7_3 {},m_7_4 {}, m_7_5 {} , m_7_6 {},m_7_7 {}, b_7 {},squareError {}".format(m_7_1,m_7_2,m_7_3,m_7_4,m_7_5,m_7_6,m_7_7,b_7,er_7))
test_er7=compute_SquareError_polynomial_7(b_7, m_7_1, m_7_2, m_7_3,m_7_4,m_7_5,m_7_6,m_7_7, x_test,y_test)
print("test error on test data for polynomial n==7 is ==",test_er7)
y_predict_poly7=[]
for i in range(len(df.x)):
    y_predict_poly7.insert(i, b_7+m_7_1*df.x[i]+m_7_2*df.x[i]*df.x[i]+m_7_3*df.x[i]*df.x[i]*df.x[i]+m_7_4*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_7_5*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_7_6*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_7_7*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=7")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly7, color='blue')
plt.show()
#polynomial n=8 ------------------
[b_8,m_8_1,m_8_2,m_8_3,m_8_4,m_8_5,m_8_6,m_8_7,m_8_8,er_8]=gradient_descent_polynomial_x_8(x,y,iteration,learning_rate)
print("For polynomial n=8 estimated W : m_8_1 {}, m_8_2 {} ,m_8_3 {},m_8_4 {}, m_8_5 {} , m_8_6 {},m_8_7 {},m_8_8 {},b_8 {},squareError {}".format(m_8_1,m_8_2,m_8_3,m_8_4,m_8_5,m_8_6,m_8_7,m_8_8,b_8,er_8))
test_er8=compute_SquareError_polynomial_8(b_8, m_8_1, m_8_2, m_8_3,m_8_4,m_8_5,m_8_6,m_8_7,m_8_8, x_test,y_test)
print("test error on test data for polynomial n==8 is ==",test_er8)
y_predict_poly8=[]
for i in range(len(df.x)):
    y_predict_poly8.insert(i, b_8+m_8_1*df.x[i]+m_8_2*df.x[i]*df.x[i]+m_8_3*df.x[i]*df.x[i]*df.x[i]+m_8_4*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_8_5*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_8_6*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_8_7*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_8_8*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=8")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly8, color='blue')
plt.show()
#polynomial n=9 ------------------
[b_9,m_9_1,m_9_2,m_9_3,m_9_4,m_9_5,m_9_6,m_9_7,m_9_8,m_9_9,er_9]=gradient_descent_polynomial_x_9(x,y,iteration,learning_rate)
print("For polynomial n=9 estimated W : m_9_1 {}, m_9_2 {} ,m_9_3 {},m_9_4 {}, m_9_5 {} , m_9_6 {},m_9_7 {},m_9_8 {},m_9_9 {},b_9 {},squareError {}".format(m_9_1,m_9_2,m_9_3,m_9_4,m_9_5,m_9_6,m_9_7,m_9_8,m_9_9,b_9,er_9))
test_er9=compute_SquareError_polynomial_9(b_9, m_9_1, m_9_2, m_9_3,m_9_4,m_9_5,m_9_6,m_9_7,m_9_8,m_9_9, x_test,y_test)
print("test error on test data for polynomial n==9 is ==",test_er9)

y_predict_poly9=[]
for i in range(len(df.x)):
    y_predict_poly9.insert(i, b_9+m_9_1*df.x[i]+m_9_2*df.x[i]*df.x[i]+m_9_3*df.x[i]*df.x[i]*df.x[i]+m_9_4*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_9_5*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_9_6*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_9_7*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_9_8*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]+m_9_9*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i]*df.x[i])
#print(y_predict_poly1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Fitted curve for x polynomial n=9")
plt.scatter(df.x,df.y, color='red', marker= '*')
plt.plot(df.x,y_predict_poly9, color='blue')
plt.show()