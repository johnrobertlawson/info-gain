"""
Custom exceptions.
"""

class UnboundedProbError(Exception):
    def __init__(self,f,o):
        if f in (0,1):
            if o in (0,1):
                v_str = "both f ({f}) and o ({o}) are bad values."
            else:
                v_str = "f = {f}."
        elif o in (0,1):
            v_str = "o = {o}."
        else:
            # Should only use the error if there's an error!
            raise Exception
        print("Make sure both forecast and observation are not binary "
                "values. Currently,",v_str)
        return

class SillyValueError(Exception):
    print("The given values are nonsensical.")