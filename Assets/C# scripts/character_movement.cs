using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class character_movement : MonoBehaviour
{
    // private Transform transform;

    private SpriteRenderer spriteRendererComponent;

    private float moveSpeed = 5f;
    private bool canMove = true;

    // To determine which direction projectiles are shooting at based on the direction the player is facing
    // in the seperate 'firing_projectile.cs' C# script. 
    public bool isfacingright = true;      // if this variable is false, then the player is facing left

    // Start is called before the first frame update
    void Start()
    {
        // Get the SpriteRenderer component from the GameObject
        spriteRendererComponent = GetComponent<SpriteRenderer>();

        // Check if the SpriteRenderer component is not null
        if (spriteRendererComponent == null)
        {
            Debug.LogError("SpriteRenderer component not found on the GameObject.");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (canMove)
        {
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                transform.Translate(-Vector3.right * moveSpeed * Time.deltaTime);
                FlipSprite(true); // Dont flip horizontally
                isfacingright = false;
            }
            if (Input.GetKey(KeyCode.RightArrow))
            {
                transform.Translate(Vector3.right * moveSpeed * Time.deltaTime);
                FlipSprite(false); // Dont flip horizontally
                isfacingright = true;
            }
        }
    }

    // Method to toggle movement on/off (for in the 'interaction' C# scripts)
    public void ToggleMovement(bool allowMovement)
    {
        canMove = allowMovement;
    }

    // Method to flip the sprite horizontally
    private void FlipSprite(bool flip)
    {
        // Flip the sprite horizontally based on the flip parameter. 'flipX' is a property of the 'SpriteRenderer' 
        // component in Unity, and it is built-in that flips the sprite image in the horizontal x-axis
        spriteRendererComponent.flipX = flip;
    }
}
