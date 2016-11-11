class Rectangle:
    """ A class to manufacture rectangle objects """

    def __init__(self, x, y , w, h):
        """ Initialize rectangle at posn, with width w, height h """
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def __str__(self):
        return  "(x:{0}, y:{1}, w:{2},h:{3})".format(self.x,self.y, self.width, self.height)

    def intersect(self,rect):
        if (self.x + self.width < rect.x or rect.x + rect.width < self.x or self.y + self.height < rect.y or rect.y + rect.height < self.y):
            return False
        else:
            return True

def intersect_collection(rect,collection):
    for rect1 in collection:
        if rect.intersect(rect1):
            return True

    return False
