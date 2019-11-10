# Easing Animations with Python
This is a set of functions that allow you to bring After effects style easing to your python animation by interpolatating a set of parametric arrays.

Ingredients necessary to make the animations:
* an intitial time vector
* a dependent data matrix - each row will be a different variable, the columns correspond to the initial time vector
* an output time vecotr upon which to be interpolated

# Interpolation class
Before generating the easing, you must create the object of class easing
```python
    ease = Eased(data, input_time_vector, output_time_vector)
```
All subsequent functions will be called on this object.

 ![traces](media/interpolation_schema.png)

# Power Interpolation

The primary form of interpolative easing (or *tweening*) is based on powers (e.g. linear, quadratic, cubic etc.)
 The power easing function takes one variable - the exponent integer (n). Increasing the interger
 increases the *sharpness* of the transition.

 ```python
    out_data = ease.power_ease(n)
```
 ![traces](media/traces.png)

 ![Demo](media/comparison.gif)

# No interpolation
If you simply want to extend the data to have the same number of points as an interpolated set
without actually interpolating, simply call the No_interp() function
```python
    out_data = ease.No_interp()
```

<!-- 

# To do list
* meta ask how do we track these projects?
options include:
-jira
-trello
-[meat space notepads]

* hit major objectives for MVP
-cleanup and remererge 
-save files to scratch, not in the repo
-2d scatterplot
-evolving histogram
-some sort of callable example interface?
-


* nice haves
-chaginger trendline?
-flashy example (chroma shift)
-map overlay?
-bending curve  (not scatter plot) -->
