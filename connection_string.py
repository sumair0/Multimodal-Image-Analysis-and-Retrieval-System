username = 'dverhage'
password = 'CdV51J7gRmAxsBOj'
dbname = 'phase2'
cluster = 'cluster0'
def connection_string():
    connection_string = f'mongodb+srv://{username}:{password}@{cluster}.giplf.mongodb.net/{dbname}?retryWrites=true&w=majority'
    return connection_string