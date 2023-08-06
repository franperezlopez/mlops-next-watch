import kafka

kafka_broker = kafka.KafkaAdminClient(bootstrap_servers='kafka:9092', api_version=(3, 5, 1))

topics = kafka_broker.list_topics() #.topics

print(topics)

if not topics:
    raise RuntimeError()


