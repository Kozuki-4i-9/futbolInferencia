import numpy as np

def decorador_prueba_0(func):
  dic_emparejamientos = {"1986":({"diccionario 0"}, {"diccionario 1"}, np.array([]), [[],[]], []),
                         "1990":({"diccionario 0"}, {"diccionario 1"}, np.array([]), [[],[]], []),
                         "1994":({"diccionario 0"}, {"diccionario 1"}, np.array([]), [[],[]], []),
                         "2026":({"diccionario 0"}, {"diccionario 1"}, np.array([]), [[],[]], [])}

  def wrapper(self, *args, **kwargs):

    effective_clave_0 = kwargs.get("clave_0", "no")
    if effective_clave_0 == "vacio":
      var0 = dic_emparejamientos["1986"]
      kwargs["clave_0"] = var0

    return func(self, *args, **kwargs)

  return(wrapper)

class clase_prueba_0(object):
  @decorador_prueba_0
  def __init__(self, clave_0="vacio"):
    self.clave_0 = clave_0

class clase_prueba_1(clase_prueba_0):
  def __init__(self):

    super().__init__(clave_0="vacio")
    self.dic_0 = self.clave_0[0]
    self.dic_1 = self.clave_0[1]

    self.array_0 = self.clave_0[2]

    self.listas_0 = self.clave_0[3]
    self.lista_1 = self.clave_0[4]

objeto0 = clase_prueba_1()
