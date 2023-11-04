package com.example.ai_project2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.view.View;
import android.widget.Button;

public class Profile extends AppCompatActivity {

    private Button button_start;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_profile);
        button_start = findViewById(R.id.start);

        button_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
             toMain();
            }
        });


    }
    public void toMain(){
        Intent intent = new Intent(this,MainActivity.class);
        startActivity(intent);

    }
}