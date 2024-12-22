#dict constructor
thisdict = dict(name = "John", age = 36, country = "Norway")
print(thisdict)

#check if key exists
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
if "model" in thisdict:
  print("Yes, 'model' is one of the keys in the thisdict dictionary")

#update the value with update method or u can add key:value pair to the end of dict using update
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict.update({"year": 2020})
thisdict.update({"color": "red"})

#pop() removes the item with specified key  ( .popitem() will remove the last key:value
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict.pop("model")
print(thisdict)

#or u can use del to delete the item with the specified key name
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
del thisdict["model"]
print(thisdict)

#loop trough keys
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
for x in thisdict:
  print(x)

#Print all values in the dictionary, one by one: (or use thisdict.values()/keys()
for x in thisdict:
  print(thisdict[x])

#to copy use .copy()  or dict()
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = thisdict.copy()
print(mydict)

mydict = dict(thisdict)
print(mydict)

#access to nested dicts
myfamily = {
  "child1" : {
    "name" : "Emil",
    "year" : 2004
  },
  "child2" : {
    "name" : "Tobias",
    "year" : 2007
  },
  "child3" : {
    "name" : "Linus",
    "year" : 2011
  }
}

print(myfamily["child2"]["name"])

#loop through nested loops
for x, obj in myfamily.items():
  print(x)

  for y in obj:
    print(y + ':', obj[y])

#defaultdict is a special type of dictionary in Python, provided by the collections module. It behaves like a normal dictionary, but with one key difference: it provides a default value for a non-existent key without raising a KeyError.
from collections import defaultdict

#default value = 0
d = defaultdict(int)
d["a"] += 1  # Key "a" doesn't exist, so it gets the default value 0, then 1 is added
print(d)  # Output: defaultdict(<class 'int'>, {'a': 1})

#default value = []
d = defaultdict(list)
d["a"].append(1)  # Key "a" doesn't exist, so it gets the default value []
print(d)  # Output: defaultdict(<class 'list'>, {'a': [1]})
#we use append cause add to empty list value

#USAGE
from collections import defaultdict

counts = defaultdict(int)
items = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']

for item in items:
    counts[item] += 1

print(counts)  # Output: defaultdict(<class 'int'>, {'apple': 3, 'banana': 2, 'orange': 1})

#grouping data
from collections import defaultdict

groups = defaultdict(list)
data = [("fruit", "apple"), ("fruit", "banana"), ("veg", "carrot")]

for category, value in data:
    groups[category].append(value)

print(groups)  # Output: defaultdict(<class 'list'>, {'fruit': ['apple', 'banana'], 'veg': ['carrot']})
