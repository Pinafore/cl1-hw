
prices = {'apples': 2.0, 'oranges': 1.5, 'pears': 1.75}

for fruit, price in prices.items():
    if price < 2.0:
        print ("I'll buy %s" % fruit)
    else:
        print ("%s is too expensive" % fruit)
