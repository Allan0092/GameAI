using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class character_gravity_physics : MonoBehaviour
{
    // Reference to the Rigidbody component
    private Rigidbody2D rigidbody2Dcomponent;

    // Start is called before the first frame update
    void Start()
    {
        // Get the Rigidbody component attached to the character
        rigidbody2Dcomponent = GetComponent<Rigidbody2D>();
    }


    // Update is called once per frame
    void Update()
    {

    }
}
