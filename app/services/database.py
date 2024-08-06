from firebase_admin import firestore


def create_document(collection, doc_id, data):
    """
    Creates a new document in the specified collection with the given document ID and data.

    Args:
        collection (str): The name of the collection to create the document in.
        doc_id (str): The ID of the document to be created.
        data (dict): The data to be stored in the document.
    """
    db = firestore.client()
    return db.collection(collection).document(doc_id).set(data)


def get_documents(collection, field, op, value):
    """
    Retrieves documents from a Firestore collection based on the provided field, operator, and value.

    Args:
        collection (str): The name of the Firestore collection.
        field (str): The field to filter the documents by.
        op (str): The operator to use for the filtering operation.
        value: The value to compare against the field.
    """
    db = firestore.client()

    # Check if field, op, and value are not None
    if field is not None and op is not None and value is not None:
        # Filter documents based on the provided field, operator, and value
        return db.collection(collection).where(field, op, value).stream()

    # Return all documents from the collection
    return db.collection(collection).stream()


def get_document(collection, doc_id):
    """
    Retrieves a document from the specified collection in the Firestore database.

    Args:
        collection (str): The name of the collection to retrieve the document from.
        doc_id (str): The ID of the document to retrieve.
    """
    db = firestore.client()
    return db.collection(collection).document(doc_id).get()


def delete_document(collection, doc_id):
    """
    Deletes a document from the specified collection in the Firestore database.

    Args:
        collection (str): The name of the collection where the document is located.
        doc_id (str): The ID of the document to be deleted.
    """
    db = firestore.client()
    return db.collection(collection).document(doc_id).delete()


def update_document(collection, doc_id, data):
    """
    Update a document in the specified collection with the given data.

    Args:
        collection (str): The name of the collection.
        doc_id (str): The ID of the document to be updated.
        data (dict): The data to be updated in the document.
    """
    db = firestore.client()
    return db.collection(collection).document(doc_id).update(data)
