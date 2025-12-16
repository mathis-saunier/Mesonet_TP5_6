import pickle
import numpy as np

def inspect_data(filename):
    print(f"Inspecting {filename}...")
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            # x_train, yy_train, x_test, yy_test
            if len(data) == 4:
                x_train, yy_train, x_test, yy_test = data
                print(f"  Train size: {len(yy_train)}")
                print(f"  Test size: {len(yy_test)}")
                
                # Check labels for orientation tokens (10 and 11)
                # yy_train is likely numpy array of shape (N, 6) or similar?
                # Based on main_transformer.py: yy_train[n].T[0][:]
                
                unique_tokens = set()
                for i in range(len(yy_train)):
                    labels = yy_train[i].flatten()
                    unique_tokens.update(labels)
                
                print(f"  All tokens found: {unique_tokens}")
                if 10 in unique_tokens: print("  Contains Horizontal (10)")
                if 11 in unique_tokens: print("  Contains Vertical (11)")
            else:
                print("  Unexpected data structure")
    except Exception as e:
        print(f"  Error: {e}")

inspect_data('MNIST_5digits2DHorizontalFacile.pkl')
inspect_data('MNIST_5digitsDifficile.pkl')
inspect_data('MNIST_5digits2DHorizontal.pkl')
inspect_data('MNIST_5digits2D.pkl')
