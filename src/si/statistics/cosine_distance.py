import numpy as np


def cosine_distance(x:np.ndarray, y:np.ndarray) -> float:
    
    """
    Calculates the Cosine distance between X and Y using the following formula: similarity(x,y)= x⋅y/∣∣x∣∣∣∣y∣∣  
    where x⋅y is the dot product of the vectors x and y and ∣∣x∣∣∣∣y∣∣ is the product of the Euclidean norms of x and y.
 
    Parameters
    ----------
    x: np.array
        A numpy array
    y: np.array
        A numpy array
        
    Returns
    ----------
    cosine_distance: float
        The cosine distance between X and Y
    
    """
    distances= []
    dot_product = np.dot(x, y) #faz a multiplicação 
    magnitude_x = np.linalg.norm(x)
    magnitude_y = np.linalg.norm(y)
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    cosine_distance = 1 - cosine_similarity
    distances.append(cosine_distance)
    
    return distances