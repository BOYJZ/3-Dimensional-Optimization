import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def noisy_gaussian(x, y, z, SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C):
  f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)+ (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
  noisy_f =  A * f + A * np.random.normal(0, (1/SNR), f.shape) + C
  return noisy_f

'''
num_samples: number of samples
num_points: number of points in each samples
    [lower_snr,higher_snr]: range of SNR of each samples
center: detect center
input_snr, input_location: for the 4 last experiments, use same snr and location as first generation
'''
def generate_dataset(num_samples,num_points,lower_snr,higher_snr):
    X = np.zeros((num_samples, num_points))
    y = np.zeros((num_samples, 3))
    array_A=[]
    array_B=[]
    snr=[]
    for i in tqdm(range(num_samples)):
        A=np.random.randint(3000, 493000)
        B=np.random.randint(300,7000)
        C=B
        
        #SNR = np.random.uniform(lower_snr, higher_snr)
        SNR = abs(np.random.normal(0, std_dev))

        #maximum location
        mu_x = np.random.uniform(-1, 1)
        mu_y = np.random.uniform(-1, 1)
        mu_z = np.random.uniform(-1, 1)
        
        #sigma_x = np.random.uniform(0.5, 2)
        #sigma_y = np.random.uniform(0.5, 2)
        #sigma_z = np.random.uniform(0.5, 2)
        sigma_x,sigma_y,sigma_z=1,1,1

        points_per_dim = int(np.ceil(num_points**(1/3.)))

        dx = np.linspace(-1, 1, points_per_dim)
        dy = np.linspace(-1, 1, points_per_dim)
        dz = np.linspace(-1, 1, points_per_dim)

        xx, yy, zz = np.meshgrid(dx, dy, dz)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:num_points]

        values=[]
        values = noisy_gaussian(points[:,0], points[:,1], points[:,2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C)
        max_value=max(values)
        normalized_values = [x / max_value for x in values]
        X[i] = np.array(normalized_values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        
        #save
        array_A.append(A)
        array_B.append(B)
        snr.append(SNR)
    return np.array(X), np.array(y), np.array(snr), array_A, array_B

def generate_dataset_pro(num_samples,num_points,lower_snr,higher_snr):
    X = np.zeros((num_samples, num_points))
    y = np.zeros((num_samples, 3))
    array_A=[]
    array_B=[]
    snr=[]
    for i in tqdm(range(num_samples)):
        A=np.random.randint(3000, 493000)
        B=np.random.randint(300,7000)
        C=B
        
        SNR = np.random.uniform(lower_snr, higher_snr)
        #SNR = abs(np.random.normal(0, std_dev))

        #maximum location
        mu_x = np.random.uniform(-1, 1)
        mu_y = np.random.uniform(-1, 1)
        mu_z = np.random.uniform(-1, 1)
        
        #sigma_x = np.random.uniform(0.5, 2)
        #sigma_y = np.random.uniform(0.5, 2)
        #sigma_z = np.random.uniform(0.5, 2)
        sigma_x,sigma_y,sigma_z=1,1,1

        points_per_dim = int(np.ceil(num_points**(1/3.)))

        dx = np.linspace(-1, 1, points_per_dim)
        dy = np.linspace(-1, 1, points_per_dim)
        dz = np.linspace(-1, 1, points_per_dim)

        xx, yy, zz = np.meshgrid(dx, dy, dz)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:num_points]

        values=[]
        values = noisy_gaussian(points[:,0], points[:,1], points[:,2], SNR, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z,A,B,C)
        max_value=max(values)
        normalized_values = [x / max_value for x in values]
        X[i] = np.array(normalized_values).T
        y[i] = np.array([mu_x, mu_y, mu_z])
        
        #save
        array_A.append(A)
        array_B.append(B)
        snr.append(SNR)
    return np.array(X), np.array(y), np.array(snr), array_A, array_B

########################################################################################
num_samples=30000000
num_points=16
lower_snr=1
higher_snr=100
std_dev = 35
measurements=200
num_test=10000
train_epochs=3
########################################################################################

# Generate the dataset
dataset_path = 'C:\\Users\\jincheng\\桌面\\Auto-focusing\\3e7&16points.npz'
if not os.path.exists(dataset_path):
    
    print('not exist')
    X_train, y_train, snr_train, array_A_train, array_B_train = generate_dataset(num_samples,num_points,lower_snr,higher_snr)
    np.savez(dataset_path, X_train=X_train, y_train=y_train, snr_train=snr_train, array_A_train=array_A_train, array_B_train=array_B_train)
else:
    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    snr_train = data['snr_train']
    array_A_train = data['array_A_train']
    array_B_train = data['array_B_train']

# Reshape the input data to match the expected input shape of the model
X_train_input = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Define the input shape
input_shape = (num_points, 1)

model = models.Sequential()
model.add(layers.Conv1D(6, (5,), activation='relu', input_shape=(num_points, 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(16, (5,), activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(3, activation='linear'))  # regression task

# Compile the model
model.compile(optimizer='Adam', loss='MSE', metrics=['mean_absolute_error'])

#fit the model
model.fit(X_train_input, y_train, epochs=train_epochs)

#save the model
model.save('my_model.h5')

#how to load model?
#from keras.models import load_model
#model = load_model('my_model.h5')

def draw_pro_snr():   
    lower_snr=1
    higher_snr=100
    X_test, y_test,snr,  array_A, array_B = generate_dataset_pro(num_test,num_points,lower_snr,higher_snr)
    y_pred = model.predict(X_test)
    success=0
    x_snr=[]
    y_pro=[]
    total_number=[]
    num_parts=50
    for i in range(num_parts):#split the snr into 5 parts, convert continuous snr to discrete snr
        x_snr.append(lower_snr+(higher_snr-lower_snr)*i/(num_parts-1))
        y_pro.append(0)
        total_number.append(0)
    for i in range(len(y_pred)):
        distance = np.sqrt(  (y_pred[i][0]-y_test[i][0])**2 + (y_pred[i][1]-y_test[i][1])**2 + (y_pred[i][2]-y_test[i][2])**2  )
        tem_snr = min(x_snr, key=lambda x: abs(x-snr[i]))
        #print(int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) ))
        total_number[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
        #print('distance',distance)
        #print('test',y_test[i])
        #print('prediction',y_pred[i])
        #print('#####################################################################')
        if distance < 0.44:
            success+=1
            y_pro[int( (tem_snr-lower_snr) / ((higher_snr-lower_snr)/(num_parts-1)) )]+=1
    for i in range(num_parts):
        if total_number[i]==0:
            continue
        y_pro[i]=y_pro[i]/total_number[i]        
    print(success/len(y_pred))
    print('discrete snr is',x_snr)
    print('#success in each snr is',y_pro)
    print('#total examples in each snr is',total_number)
    print('#############################################################################')
    plt.plot(x_snr,y_pro)
    plt.xlabel('discrete snr')
    plt.ylabel('pro of success')
    plt.title('Success Probability vs SNR')
    plt.show()
    return

draw_pro_snr()