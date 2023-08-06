import kafka
import pickle

producer = kafka.KafkaProducer(bootstrap_servers='kafka:9092', api_version=(3, 5, 1))

print("hey")
#record = kafka.ProducerRecord('teste', b'my-message', delivery_mode=1)
sdata = pickle.dumps(['my-message', 1, 1.2])
producer.send('teste', sdata)
print("yo")

producer.flush()
print("aaa")
