using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class enemy_follow_player : MonoBehaviour
{
    public string playerName = "bird character sprite";
    private GameObject player;
    public float moveSpeed = 1f; // Speed at which the enemy moves

    private Rigidbody2D rb; // Reference to a 'RigidBody' component

    public float interactionRange = 5f;
    private float distance;
    private bool startfollowing; 
    

    // Start is called before the first frame update
    void Start()
    {
        startfollowing = false;

        player = GameObject.Find(playerName);
        rb = GetComponent<Rigidbody2D>();
    }


    // Update is called once per frame
    void Update()
    {
        // Calculate the direction from the enemy to the player
        Vector2 directionToPlayer = player.transform.position - transform.position;

        // Calculate the distance continuously
        distance = directionToPlayer.magnitude;


        // Setting a trigger range, when, upon entering, triggers the enemy NPC to start following the character

        // Check if the player is within the interaction range
        if (distance <= interactionRange)
        {
            startfollowing = true;
        }

        if (player != null && startfollowing)
        {
            // Normalize the direction to get a unit vector
            directionToPlayer.Normalize();

            // Move the enemy using Rigidbody2D
            rb.linearVelocity = directionToPlayer * moveSpeed;
        }
    }
}
