import Inverted_pendulum.data as data_ip
import Inverted_pendulum.prob as prob_ip
import Double_integrator.data as data_di
import Double_integrator.prob as prob_di
import unicycle_model.data as data_uni
import unicycle_model.prob as prob_uni


def system_data(sys):
     if sys == 'ip':
          data = data_ip
          prob = prob_ip
     elif sys == 'di':
          data = data_di
          prob = prob_di
     elif sys == 'uni':
          data = data_uni
          prob = prob_uni

     return data, prob
