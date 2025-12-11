## 2025-12-10

One of the main design decisions right now is whether to integrate SEG stuff into the backend.

Pros:
* probably more efficient
* might be the only way this is possible: e.g. how do we correctly intercept the case where rules add new rows to the database? in those cases we need to fire the semantic constructor. how do we intercept the add, though? seems like it'd need to be in the backend.

cons:
* might not be possible without significant modification: type information is not possible in the backend. 