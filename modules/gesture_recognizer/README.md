# Gesture Recognizer  

## Aim  
The aim of the gesture recognizer is to compare two gestures and decide if both of them are same.  
  
## Hand Landmark Locations  
  
```
	        8   12  16  20  
	        |   |   |   |  
	        7   11  15  19  
	    4   |   |   |   |  
	    |   6   10  14  18  
	    3   |   |   |   |  
	    |   5---9---13--17  
	    2    \         /  
	     \    \       /  
	      1    \     /  
	       \    \   /  
	        ------0-  
```  
  
## Implementation Details  
  
The gesture landmark are just the points on the 2d plane with (x,y) coordinates for each point.  
  
The first thing we do is ask the user to define a lock gesture and we save this gesture's landmark which will be used to compare with new landmarks while unlocking.  
  
The simplest way to compare these landmarks could be to take Euclidean distance of all corresponding points but this method was very bad in prediction because the user can place their hand at any place and orientation which results in unpredictably large errors.  
  
In these type of cases the input could vary with different Orientation, Rotation, translation, scale. So I had to find a better method to compare gestures while reducing translation, rotation and scale variations so that we could easily compare them.  
  
This type of problem is also referred as [Procrustes superimposition](https://en.wikipedia.org/wiki/Procrustes_analysis) (PS) which is performed by optimally translating, rotating and uniformly scaling the objects. In other words, both the placement in space and the size of the objects are freely adjusted. The aim is to obtain a similar placement and size, by minimizing a measure of shape difference called the Procrustes distance between the objects. This is sometimes called full, as opposed to partial PS, in which scaling is not performed (i.e. the size of the objects is preserved). Notice that, after full PS, the objects will exactly coincide if their shape is identical.   
  
![Procrustes Superimposition Example](https://upload.wikimedia.org/wikipedia/commons/f/f5/Procrustes_superimposition.png)

## References

- code - https://stackoverflow.com/a/18927641
- https://en.wikipedia.org/wiki/Procrustes_analysis
