from collections import namedtuple

class TaggedVector(namedtuple('TaggedVector', 'vectors tags')):
    def __str__(self):
        """Human readable representation of the object's state, used for debugging.
        Returns
        -------
        str
           Human readable representation of the object's state (vectors and tags).
        """
        return '%s<%s, %s>' % (self.__class__.__name__, self.vectors, self.tags)